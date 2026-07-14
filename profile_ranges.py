#!/usr/bin/env python3
"""Phase 3 (READ): per-tensor range profiling for the activation-quant design (Table 3).

This JSON is what Table 3 (per-layer ap_fixed<B,I>) is built from, so every number is
PER LAYER-INSTANCE / PER CLIP-SITE-x-INSTANCE, not aggregated across the network.

WHAT IS MEASURED
  * Layer OUTPUTS: every custom-layer instance's output, by WRAPPING its .call
    (embedders, CNNKernelWithEmbedding, CNNStateCorrelator, RCNNKernelCombiner,
    StateDecoder + inner Dense/QDense). Keyed per instance (#idx).
  * PRE-CLIP inputs: clip_zlike / clip_exp wrapped, keyed by CALL-SITE **and INSTANCE**
    (file:line#Class#idx) via the calling frame's `self`. Records range, frac_clipped,
    and (for the exp/combination clip) frac_nonpositive.

NAMING: clip_exp clips the COMBINATION that goes into log(): res = Sum_i x_i +
2 Sum_{i<j} c_phi,ij sqrt(x_i x_j), with c_phi in (-1,1). x_i = e^z > 0, but the
phase-weighted cross terms are SIGNED, so the combination can be <= 0 (clip_exp floors
it before log). So the tag is `combination_preclip`, NOT exp_preclip, and the finding is
"the phase-weighted COMBINATION can be <=0" -> log-domain rewrite needs SIGNED LSE.

READ FIRST: combination_preclip.eff_z span + frac_nonpositive, AND any layer output with
implied_frac_at_B6 < 0 (x-like, un-representable at sane B). Both decide the LSE rewrite.

signed comes from the variable TYPE where known (clip sites -> signed; decoder ReLU ->
unsigned; sigmoid -> unsigned; else inferred from data and LABELED). An unsigned-typed
site that goes negative is a TYPE_VIOLATION (flagged), not silently absorbed.

Run PER SEED (weights filename carries seedN -> output profile_ranges_w{B}_seed{N}.json).
If implied_I moves across seeds, that's a Table 3 caveat.

  .venv/bin/python profile_ranges.py --weights <w6 seedN .weights.h5> [--expect-pl 0.046675]
"""
import argparse, json, math, os, re, sys
import numpy as np

_CAP = {}
_HOOKED = set()
_INSTANCE_IDX = {}         # id(layer_instance) -> "ClassName#idx"
_SUBSAMPLE = 20000
_rng = np.random.default_rng(0)


def _acc(site, arr, clip=None, vclass=None):
    import tensorflow as tf
    assert tf.executing_eagerly(), f'capture at {site} ran under tracing (not eager)'
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return
    s = _CAP.setdefault(site, dict(n=0, lo=math.inf, hi=-math.inf, minnz=math.inf,
                                   sum=0.0, sample=[], n_clip=0, clip=None, vclass=vclass))
    s['vclass'] = vclass if vclass is not None else s.get('vclass')
    s['n'] += a.size
    s['lo'] = min(s['lo'], float(a.min()))
    s['hi'] = max(s['hi'], float(a.max()))
    av = np.abs(a); nz = av[av > 0]
    if nz.size:
        s['minnz'] = min(s['minnz'], float(nz.min()))
    s['sum'] += float(a.sum())
    if clip is not None:
        lo, hi = clip
        s['clip'] = (lo, hi)
        if lo > 0:   # exp/combination clip: non-positive is a FINDING (ln undefined -> signed LSE)
            nonpos = a <= 0
            s['n_nonpos'] = s.get('n_nonpos', 0) + int(nonpos.sum())
            s['n_clip'] += int(((a < lo) & ~nonpos).sum()) + int((a > hi).sum())
        else:
            s['n_clip'] += int(((a < lo) | (a > hi)).sum())
    idx = _rng.integers(0, a.size, _SUBSAMPLE) if a.size > _SUBSAMPLE else np.arange(a.size)
    s['sample'].append(a[idx].astype(np.float32))


def _rec_out(tag, out, vclass=None):
    if isinstance(out, (list, tuple)):
        for i, o in enumerate(out):
            _rec_out(f'{tag}[{i}]', o, vclass)
        return
    if hasattr(out, 'shape'):
        if not hasattr(out, 'numpy'):
            raise RuntimeError(f'{tag} produced a symbolic tensor -- capture ran under tracing')
        _acc(tag, out, vclass=vclass)


def _int_bits(v):
    v = abs(float(v))
    return 0 if v <= 0 else max(0, int(math.floor(math.log2(v))) + 1)


def _finalize(site):
    s = _CAP[site]
    vals = np.concatenate(s['sample']) if s['sample'] else np.array([0.0])
    av = np.abs(vals); nz = av[av > 0]
    p99_9 = float(np.percentile(av, 99.9))
    vc = s.get('vclass')
    if vc == 'signed':
        signed, src = True, 'typed'
    elif vc == 'unsigned':
        signed, src = False, 'typed'
    else:
        signed, src = bool(s['lo'] < 0), 'inferred'
    I_max = _int_bits(max(abs(s['lo']), abs(s['hi'])))
    d = dict(
        n=s['n'], min=s['lo'], max=s['hi'], mean=s['sum'] / max(s['n'], 1),
        absmax=max(abs(s['lo']), abs(s['hi'])), abs_p99_9=p99_9,
        signed=signed, signed_source=src, vclass=vc,
        type_violation=bool(vc == 'unsigned' and s['lo'] < -1e-4),
        min_nonzero_full=(None if math.isinf(s['minnz']) else s['minnz']),
        implied_I_from_max=I_max, implied_I_from_p99_9=_int_bits(p99_9),
        implied_frac_at_B6=6 - I_max - int(signed), implied_frac_at_B8=8 - I_max - int(signed),
    )
    if s['clip'] is not None:
        d['clip'] = list(s['clip'])
        d['frac_clipped'] = s['n_clip'] / max(s['n'], 1)
        if 'n_nonpos' in s:
            d['frac_nonpositive'] = s['n_nonpos'] / max(s['n'], 1)
    denom = float(np.std(vals)) + 1e-12
    I_p999 = _int_bits(p99_9)

    def _sim(I, suffix):
        for B in (8, 6, 4):
            frac = B - I - int(signed)
            if frac < 0:
                d[f'abs_rmse_at_B{B}{suffix}'] = None; d[f'rel_rmse_at_B{B}{suffix}'] = None
                continue
            step = 2.0 ** (-frac)
            qlo = -(2.0 ** I) if signed else 0.0
            qhi = 2.0 ** I - step
            q = np.clip(np.round(vals / step) * step, qlo, qhi)
            e = float(np.sqrt(np.mean((q - vals) ** 2)))
            d[f'abs_rmse_at_B{B}{suffix}'] = e; d[f'rel_rmse_at_B{B}{suffix}'] = e / denom

    _sim(I_max, ''); _sim(I_p999, '_Ip999')
    return d


def install_clip_hooks():
    """Wrap clip_zlike/clip_exp; key by CALL-SITE + calling INSTANCE; record pre-clip + rate."""
    import utilities_arrayops as ua
    VB = ua.VariableBounds
    oz, oe = VB.clip_zlike, VB.clip_exp
    zb = float(VB.bound_zlike)
    elo, ehi = math.exp(-zb), math.exp(zb)

    def _site(prefix):
        fr = sys._getframe(2)               # 0=_site,1=wrapper,2=caller (the layer .call)
        line = fr.f_lineno
        slf = fr.f_locals.get('self')
        inst = _INSTANCE_IDX.get(id(slf), 'agg') if slf is not None else 'agg'
        return f'{prefix}@{os.path.basename(fr.f_code.co_filename)}:{line}#{inst}'

    def wz(z):
        _acc(_site('zlike_preclip'), z, clip=(-zb, zb), vclass='signed'); return oz(z)

    def we(x):
        _acc(_site('combination_preclip'), x, clip=(elo, ehi), vclass='signed'); return oe(x)

    VB.clip_zlike = staticmethod(wz)
    VB.clip_exp = staticmethod(we)
    return zb


def install_output_hooks(model):
    import CNNModel
    names = ['DetectorBitStateEmbedder', 'DetectorEventStateEmbedder', 'TripletStateProbEmbedder',
             'CNNKernelWithEmbedding', 'CNNStateCorrelator', 'RCNNKernelCombiner']
    classes = tuple(getattr(CNNModel, n) for n in names if hasattr(CNNModel, n))
    counts = {}
    for m in model.submodules:
        if isinstance(m, classes):
            cls = m.__class__.__name__
            idx = counts.get(cls, 0); counts[cls] = idx + 1
            inst = f'{cls}#{idx}'
            _INSTANCE_IDX[id(m)] = inst          # registry for the clip-hook frame lookup
            tag = f'{inst}_out'; _HOOKED.add(tag)
            oc = m.call
            def make(oc, t):
                def hooked(*a, **kw):
                    y = oc(*a, **kw); _rec_out(t, y, vclass=None); return y   # x-like: infer + label
                return hooked
            m.call = make(oc, tag)

    decoders = [m for m in model.submodules if isinstance(m, CNNModel.StateDecoder)]
    if not decoders:
        raise SystemExit('[profile] no StateDecoder in model.submodules')
    for dec in decoders:
        assert hasattr(dec, 'layers_decoder'), 'StateDecoder.layers_decoder missing'
        oc = dec.call
        def make_dec(oc):
            def hooked(inputs, *a, **kw):
                _acc('dec_in', inputs, vclass='signed'); return oc(inputs, *a, **kw)  # z-like
            return hooked
        dec.call = make_dec(oc); _HOOKED.add('dec_in')
        for i, layer in enumerate(dec.layers_decoder):
            act = getattr(getattr(layer, 'activation', None), '__name__', '')
            vclass = 'unsigned' if act in ('relu', 'sigmoid') else ('signed' if act in ('linear', '') else None)
            olc = layer.call; tag = f'dec_layer{i}_{layer.__class__.__name__}_{act or "?"}_out'; _HOOKED.add(tag)
            def make_layer(oc, t, vc):
                def hooked(x, *a, **kw):
                    y = oc(x, *a, **kw); _rec_out(t, y, vclass=vc); return y
                return hooked
            layer.call = make_layer(olc, tag, vclass)


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--weight-bits', type=int, default=6)
    ap.add_argument('--expect-pl', type=float, default=None,
                    help='sweep-recorded w6 p_L for this seed; asserts the profiled model reproduces it')
    ap.add_argument('--pl-tol', type=float, default=2e-3)
    ap.add_argument('--d', type=int, default=5); ap.add_argument('--p', type=float, default=0.010)
    ap.add_argument('--rounds', type=int, default=3); ap.add_argument('--kernel', type=int, default=3)
    ap.add_argument('--hidden', type=int, default=100); ap.add_argument('--hidden-layers', type=int, default=2)
    ap.add_argument('--npol', type=int, default=2)
    ap.add_argument('--n-test', type=int, default=200000)
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--pool', default=os.path.expanduser('~/rcnn_threshold/pools/data_d5_p0.010_r3_TAIL200k.npz'),
                    help='FRESH disjoint tail (gen-seed 43), not the old 10.01M slice')
    ap.add_argument('--out-dir', default=os.path.expanduser('~/rcnn_threshold/out_q_mcnemar'))
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    seed_m = re.search(r'seed(\d+)', os.path.basename(args.weights))
    seed = seed_m.group(1) if seed_m else 'NA'

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
    gen_seed = int(z['gen_seed']) if 'gen_seed' in z else None
    measurements = z['measurements'].astype(binary_t)
    det_evts = z['det_evts'].astype(binary_t)
    flips = z['flips'].astype(binary_t)
    det_bits, _, _ = split_measurements(measurements, d, idx_t)
    N = measurements.shape[0]; te = slice(N - nte, N)
    db, de = det_bits[te], det_evts[te]
    truth = flips[te].astype(np.int8).reshape(-1)

    zb = install_clip_hooks()
    hidden = [args.hidden for _ in range(args.hidden_layers)]
    model = build_quantized_rcnn(
        args.weight_bits, 'ZL', d, k, r, hidden, npol=args.npol, stop_round=None,
        has_nonuniform_response=False, do_all_data_qubits=False, return_all_rounds=False)
    _ = model([det_bits[0:1], det_evts[0:1]])
    model.load_weights(args.weights)
    install_output_hooks(model)
    _CAP.clear()
    global _rng
    _rng = np.random.default_rng(0)

    n = db.shape[0]; preds = np.empty(n, dtype=np.float32)
    for i in range(0, n, args.batch_size):
        j = min(i + args.batch_size, n)
        y = model([db[i:j], de[i:j]], training=False)          # eager -> per-batch capture
        preds[i:j] = np.asarray(y).reshape(-1)
        print(f'[profile] {j}/{n}', end='\r', flush=True)
    print()

    # p_L reproduction check: hooks are pure wraps -> must match the sweep's recorded w6 p_L exactly.
    pL = float(((preds > 0.5).astype(np.int8) != truth).mean())
    print(f'[profile] profiled-model p_L = {pL:.6f}'
          + (f'  (expected {args.expect_pl:.6f}, |d|={abs(pL-args.expect_pl):.2e})' if args.expect_pl else ''))
    if args.expect_pl is not None:
        assert abs(pL - args.expect_pl) <= args.pl_tol, \
            f'profiled p_L {pL:.6f} != expected {args.expect_pl:.6f} -- hooks are NOT pure wraps'

    if 'dec_in' in _CAP:
        assert _CAP['dec_in']['n'] >= n * d * d, \
            f'dec_in captured {_CAP["dec_in"]["n"]} elems, expected >= {n*d*d} -- partial capture?'
    empty = [t for t in _HOOKED if not any(kk == t or kk.startswith(t + '[') for kk in _CAP)]
    if empty:
        print(f'[profile] WARN hooked but no data (dormant at r={r}?): {sorted(empty)}')
    assert any(kk.startswith('combination_preclip') for kk in _CAP), 'no clip_exp sites fired'
    assert any(kk.startswith('zlike_preclip') for kk in _CAP), 'no clip_zlike sites fired'

    report = {site: _finalize(site) for site in sorted(_CAP)}
    comb = {s: v for s, v in report.items() if s.startswith('combination_preclip')}
    zpre = {s: v for s, v in report.items() if s.startswith('zlike_preclip')}
    di = report.get('dec_in', {})
    for s, v in comb.items():
        v['eff_z_min'] = math.log(v['min']) if v['min'] > 0 else None
        v['eff_z_max'] = math.log(v['max']) if v['max'] > 0 else None

    report['_coverage'] = dict(
        per_instance_outputs='every custom-layer instance + decoder layers (keyed #idx)',
        per_instance_clip='clip_zlike/clip_exp keyed file:line#Class#idx via calling-frame self; '
                           'sites with no self fall back to #agg (labeled)',
        combination_note='combination_preclip is the phase-weighted quadratic form into log() (signed, '
                         'can be <=0), NOT a bare exponential',
        NOT_a_phase2_qat_site=[
            'CNNStateCorrelator reverse_arg_sum (:1831) / cpwgt_arg_sum (:1856): bare tf.matmul '
            'accumulators into tanh (no Dense to wrap). Phase 2 quantizes the tanh OUTPUT '
            '(cphi/alpha, analytic I=1); accumulator width is a Phase-4 HLS decision. Deliberate scope.',
        ],
    )
    report['_provenance'] = dict(
        weights=os.path.basename(args.weights), seed=seed, pool=os.path.basename(args.pool),
        pool_gen_seed=gen_seed, weight_bits=args.weight_bits, n_test=nte,
        profiled_p_L=pL, expected_p_L=args.expect_pl,
    )
    report['_sanity'] = dict(
        zlike_bound=zb,
        dec_in_within_bound=bool(di and abs(di['min']) <= zb + 1e-3 and abs(di['max']) <= zb + 1e-3),
        zlike_preclip_absmax=max((v['absmax'] for v in zpre.values()), default=None),
        zlike_preclip_within_2x=all(v['absmax'] <= 2 * zb + 1e-3 for v in zpre.values()) if zpre else None,
        type_violations=[s for s, v in report.items() if isinstance(v, dict) and v.get('type_violation')],
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f'profile_ranges_w{args.weight_bits}_seed{seed}.json')
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[profile] wrote {out}  (weights={os.path.basename(args.weights)} gen_seed={gen_seed})')

    def _fz(x):
        return 'n/a' if x is None else f'{x:.2f}'
    print('\n=== READ FIRST (a): combination_preclip (signed LSE?  x-span decides rewrite) ===')
    for s, v in comb.items():
        print(f'  {s}: val∈[{v["min"]:.3g},{v["max"]:.3g}]  eff z∈[{_fz(v["eff_z_min"])},{_fz(v["eff_z_max"])}]  '
              f'frac_NONPOS={v.get("frac_nonpositive",0):.2%}  frac_clipped={v.get("frac_clipped",0):.2%}')
    if any(v.get('frac_nonpositive', 0) > 0 for v in comb.values()):
        print('  !! combination <=0 present -> log needs SIGNED LSE (materially harder)')

    layer_sites = [s for s in report if not (s.startswith('_') or s.startswith('combination_preclip')
                                             or s.startswith('zlike_preclip'))]
    layer_sites.sort(key=lambda s: report[s]['implied_frac_at_B6'])
    xlike = [s for s in layer_sites if report[s]['implied_frac_at_B6'] < 0]
    ok = [s for s in layer_sites if report[s]['implied_frac_at_B6'] >= 0]
    print('\n=== READ FIRST (b): UN-QUANTIZABLE layer outputs (frac@B6<0 => x-like) ===')
    for s in xlike:
        v = report[s]
        print(f'  {s:36s} max={v["max"]:+.4g}  I(max)={v["implied_I_from_max"]}  '
              f'frac@B6={v["implied_frac_at_B6"]}  -> Phase 4')
    if not xlike:
        print('  (none representable at B<=8 -> LSE rewrite may be OPTIONAL)')

    print('\n=== quantizable sites (the quantizer config) ===')
    for s in ok:
        v = report[s]
        rr6 = v.get('rel_rmse_at_B6'); rr6 = 'n/a' if rr6 is None else f'{rr6:.1%}'
        rr6p = v.get('rel_rmse_at_B6_Ip999'); rr6p = 'n/a' if rr6p is None else f'{rr6p:.1%}'
        print(f'  {s:36s} {v["signed_source"]:8s} {"signed" if v["signed"] else "unsgn"}  '
              f'I={v["implied_I_from_max"]} frac@B6={v["implied_frac_at_B6"]}  '
              f'relRMSE@B6={rr6}(Ip99.9={rr6p})')

    print('\n=== zlike pre-clip (per instance; clip-rate) ===')
    for s, v in zpre.items():
        print(f'  {s}: |max|={v["absmax"]:.3g}  frac_clipped={v.get("frac_clipped",0):.2%}')
    sn = report['_sanity']
    print(f'\nSANITY dec_in⊆±{zb:g}:{sn["dec_in_within_bound"]}  '
          f'zlike_preclip⊆±{2*zb:g}:{sn["zlike_preclip_within_2x"]} (absmax={sn["zlike_preclip_absmax"]:.3g})  '
          f'type_violations={sn["type_violations"]}')


if __name__ == '__main__':
    run()
