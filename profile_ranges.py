#!/usr/bin/env python3
"""Phase 3 (READ): per-tensor range profiling for the activation-quant design.

Only TWO sites need measurement -- the StateDecoder ReLU hidden outputs (dec_h1_out,
dec_h2_out) -- because everything else is analytically bounded by the reference architecture's clips
(z-like |.|<=12 via clip_zlike, x-like in [e^-12, e^12] via clip_exp, p/f in (0,1),
cphi/alpha in (-1,1)). See the Phase 2 site inventory.

WHY ReLU OUTPUT, not pre-activation: quantized_relu(B, I) quantizes the ReLU OUTPUT,
which is >=0, so its range is [0, max]. I = ceil(log2(max)). The pre-activation negative
tail is irrelevant (ReLU already zeroed it). So the post-ReLU max is exactly what sets I.

WHY the w6 model, not FP32: weight quantization shifts activation distributions. The
ranges the fixed-point design must contain are the w6 ranges, not FP32's. Pass
--weight-bits 6 (default). Re-run this after activation-QAT to confirm no site saturates.

FREE SANITY CHECK: StateDecoder's INPUT is the z-like state z'' (= the combiner's
clipped log-domain output, kc_z_out). We record it as `dec_in`; if it empirically
exceeds +-12 the clip is not where the design assumes -- a bug to find BEFORE the sweep.

Output: OUT_DIR/profile_ranges_w{bits}.json  (per-site min/max/mean/|.| percentiles +
the implied quantized_relu integer bits I = max(0, ceil(log2(max)))).

  .venv/bin/python profile_ranges.py --weights <w6 .weights.h5> --pool <fresh tail>
"""
import argparse, json, math, os
import numpy as np


# global eager-capture buffer: site -> running stats accumulator
_CAP = {}


def _acc(site, arr):
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    s = _CAP.setdefault(site, dict(n=0, lo=math.inf, hi=-math.inf, sum=0.0, sum2=0.0,
                                   sample=[]))
    s['n'] += a.size
    s['lo'] = min(s['lo'], float(a.min()))
    s['hi'] = max(s['hi'], float(a.max()))
    s['sum'] += float(a.sum()); s['sum2'] += float((a * a).sum())
    if len(s['sample']) < 40:                     # cap: ~40 batches of flattened vals
        s['sample'].append(a.astype(np.float32))


def _finalize(site):
    s = _CAP[site]
    vals = np.concatenate(s['sample']) if s['sample'] else np.array([0.0])
    mean = s['sum'] / max(s['n'], 1)
    absmax = max(abs(s['lo']), abs(s['hi']))
    I_relu = max(0, int(math.ceil(math.log2(s['hi']))) if s['hi'] > 0 else 0)
    return dict(
        n=s['n'], min=s['lo'], max=s['hi'], mean=mean,
        abs_p99_9=float(np.percentile(np.abs(vals), 99.9)),
        abs_p0_1=float(np.percentile(np.abs(vals), 0.1)),
        absmax=absmax,
        implied_relu_I=I_relu,          # quantized_relu(B, I): I to cover [0, max]
    )


def install_profiling_patch():
    """Patch StateDecoder.call to eager-record dec_in + each layer output."""
    import CNNModel
    orig = CNNModel.StateDecoder.call

    def profiled_call(self, inputs):
        _acc('dec_in', inputs)                                   # z-like state (clip sanity)
        x = inputs
        for i, layer in enumerate(self.layers_decoder):
            x = layer(x)
            cls = layer.__class__.__name__
            # hidden ReLU layers are all but the last (sigmoid); tag by index
            tag = f'dec_layer{i}_{cls}'
            _acc(tag, x)
        return x

    CNNModel.StateDecoder.call = profiled_call
    return orig


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help='w6 .weights.h5 (the DEPLOYED model)')
    ap.add_argument('--weight-bits', type=int, default=6, help='must match --weights (default 6)')
    ap.add_argument('--d', type=int, default=5); ap.add_argument('--p', type=float, default=0.010)
    ap.add_argument('--rounds', type=int, default=3); ap.add_argument('--kernel', type=int, default=3)
    ap.add_argument('--hidden', type=int, default=100); ap.add_argument('--hidden-layers', type=int, default=2)
    ap.add_argument('--npol', type=int, default=2)
    ap.add_argument('--n-test', type=int, default=200000)
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--pool', default=os.path.expanduser('~/rcnn_threshold/pools/data_d5_p0.010_r3_TAIL200k.npz'))
    ap.add_argument('--out-dir', default=os.path.expanduser('~/rcnn_threshold/out_q_mcnemar'))
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')

    from types_cfg import get_types
    from circuit_partition import split_measurements
    from CNNModel_quantized import build_quantized_rcnn

    d, p, r, k, nte = args.d, args.p, args.rounds, args.kernel, args.n_test
    binary_t, time_t, idx_t, packed_t = get_types(d, r, k)
    if not os.path.exists(args.pool):
        raise SystemExit(f'[profile] MISSING pool {args.pool}')
    z = np.load(args.pool)
    measurements = z['measurements'].astype(binary_t)
    det_evts = z['det_evts'].astype(binary_t)
    det_bits, _, _ = split_measurements(measurements, d, idx_t)
    N = measurements.shape[0]
    te = slice(N - nte, N)
    db, de = det_bits[te], det_evts[te]

    install_profiling_patch()
    hidden = [args.hidden for _ in range(args.hidden_layers)]
    model = build_quantized_rcnn(
        args.weight_bits, 'ZL', d, k, r, hidden, npol=args.npol, stop_round=None,
        has_nonuniform_response=False, do_all_data_qubits=False, return_all_rounds=False)
    _ = model([det_bits[0:1], det_evts[0:1]])       # build
    model.load_weights(args.weights)

    # EAGER batched forward so the python-side capture runs per batch (not just at trace)
    n = db.shape[0]
    for i in range(0, n, args.batch_size):
        j = min(i + args.batch_size, n)
        _ = model([db[i:j], de[i:j]], training=False)
        print(f'[profile] {j}/{n}', end='\r', flush=True)
    print()

    report = {site: _finalize(site) for site in sorted(_CAP)}
    # clip sanity verdict on dec_in (z-like, must be within +-12)
    di = report.get('dec_in', {})
    zb = 12.0
    report['_sanity'] = dict(
        zlike_bound=zb,
        dec_in_within_bound=bool(abs(di.get('min', 0)) <= zb + 1e-3 and abs(di.get('max', 0)) <= zb + 1e-3),
        note='if dec_in exceeds +-12 the clip_zlike is not where the design assumes',
    )
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f'profile_ranges_w{args.weight_bits}.json')
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[profile] wrote {out}')
    for site, s in report.items():
        if site.startswith('_'):
            continue
        print(f'  {site:28s} min={s["min"]:+.3f}  max={s["max"]:+.3f}  '
              f'|p99.9|={s["abs_p99_9"]:.3f}  reluI={s["implied_relu_I"]}')
    print(f'  SANITY dec_in within +-{zb}: {report["_sanity"]["dec_in_within_bound"]}')


if __name__ == '__main__':
    run()
