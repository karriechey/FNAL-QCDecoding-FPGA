#!/usr/bin/env python3
# Created: 2026-08-07
# Last updated: 2026-08-07
"""Exp 10 preflight: prove d=9, r=9 is trainable before spending ~140 GPU-hours on it.

The distance-scaling study changes exactly two things relative to the established
d=5, r=5 comparison -- distance 5 -> 9 and rounds 5 -> 9 -- and lets every tensor shape
follow from those. That is a cheap change to write and an expensive one to get subtly
wrong: a GRU whose time axis silently became the feature axis still trains and still
reports a plausible p_L, while answering a different question. So nothing launches until
every check below passes.

Two groups of checks.

Dataset (only if --pool is given; skipped otherwise so the model checks can run on a
laptop before the 14.7 GB pool exists):
  shapes and widths against the analytic values for (d, rounds)
  the three partitions are disjoint and inside the pool
  logical flip base rate on each partition
  resident memory consumed by loading the pool, the likeliest reason a d=9 run dies
  on the pod

Model, for each of RCNN / MLP / GRU:
  1. instantiate
  2. print the summary
  3. record model.count_params()
  4. push one batch through
  5. check the output shape
  6. short training smoke test
  7. confirm the loss decreased
  8. confirm validation scoring runs
  9. confirm a reloaded checkpoint predicts bit-identically

Check 9 catches a mis-saved custom model: FullRCNNModel is subclassed, so a weight file
that loads without error can still restore a different model if the build order changed.

  python preflight_d9.py                                  # models only, synthetic data
  python preflight_d9.py --pool ~/rcnn_threshold/pools_d9/data_d9_p0.010_r9_FORMAL.npz
  python preflight_d9.py --d 5 --rounds 5                 # sanity: reproduce the d=5 side
"""
import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

# The RCNN teacher's fixed configuration, unchanged from the d=5 comparison. These are
# fixed -- Exp 10 asks how the existing families scale, so retuning an architecture in
# response to d would be a bug.
KERNEL = 3
HIDDEN = [100, 100]
NPOL = 2
MLP_HIDDEN = (209, 209)
GRU_UNITS = 140


def human_bytes(n):
    return f"{n / 1024 ** 3:.2f} GB"


def rss_bytes():
    """Resident set size of this process, or None where it cannot be read cheaply."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS reports bytes.
        return peak * 1024 if sys.platform.startswith('linux') else peak
    except Exception:  # noqa: BLE001 - diagnostics only; a miss here stays silent
        return None


def check_dataset(args, fail):
    """Verify the pool's shapes, partitions and class balance. Returns a summary dict."""
    d, r = args.d, args.rounds
    n_meas_expected = r * (d ** 2 - 1) + d ** 2
    n_detbits_expected = r * (d ** 2 - 1)

    print(f"\n=== dataset: {args.pool} ===", flush=True)
    t0 = time.time()
    z = np.load(args.pool)
    measurements = z['measurements']
    det_evts = z['det_evts']
    flips = z['flips']
    n_total = measurements.shape[0]
    print(f"  loaded in {time.time() - t0:.0f}s   "
          f"peak RSS so far {human_bytes(rss_bytes() or 0)}", flush=True)

    print(f"  measurements {measurements.shape} {measurements.dtype}")
    print(f"  det_evts     {det_evts.shape} {det_evts.dtype}")
    print(f"  flips        {flips.shape} {flips.dtype}")

    if measurements.shape[1] != n_meas_expected:
        fail(f"measurement width {measurements.shape[1]} != r*(d^2-1)+d^2 "
             f"= {n_meas_expected} for d={d}, r={r}")
    # det_evts comes straight from stim's detector list, so compare against the circuit
    # itself and a change in the stim template is caught here.
    from eval_on_tail import build_circuit
    circ = build_circuit(d, args.p, r)
    if det_evts.shape[1] != circ.num_detectors:
        fail(f"det_evts width {det_evts.shape[1]} != circuit detectors "
             f"{circ.num_detectors}")
    print(f"  detector count {circ.num_detectors} (matches the circuit)")
    print(f"  det_bits width will be {n_detbits_expected}")

    # Partitions, as laid out by gen_pool_r5.py: train prefix, then validation, then the
    # sealed test block at the very end.
    tr = (0, args.n_train)
    va = (args.n_train, args.n_train + args.n_val)
    te = (n_total - args.n_test, n_total)
    print(f"  partitions: train [{tr[0]:,}, {tr[1]:,})  "
          f"validation [{va[0]:,}, {va[1]:,})  test sealed [{te[0]:,}, {te[1]:,})")

    if va[1] > n_total:
        fail(f"validation ends at {va[1]:,}, past the pool's {n_total:,} shots")
    if tr[1] > va[0]:
        fail(f"training [{tr[0]:,}, {tr[1]:,}) overlaps validation at {va[0]:,}")
    if va[1] > te[0]:
        fail(f"validation [{va[0]:,}, {va[1]:,}) overlaps the sealed test at {te[0]:,}")
    if tr[1] > te[0]:
        fail(f"training [{tr[0]:,}, {tr[1]:,}) overlaps the sealed test at {te[0]:,}")
    print("  partitions disjoint: ok")

    # Class balance. The all-zero predictor's error rate is the floor every model must
    # beat; at d=9, p=0.010 it sits near 0.49, so a collapsed model looks almost as good
    # as a working one on accuracy alone. Print it so the number is on the record.
    base_tr = float(np.asarray(flips[tr[0]:min(tr[1], tr[0] + 1000000)]).mean())
    base_va = float(np.asarray(flips[va[0]:va[1]]).mean())
    print(f"  logical flip base rate: train(first 1M) {base_tr:.4f}  "
          f"validation {base_va:.4f}")
    print("  test partition left sealed")

    fp = args.pool.replace('.npz', '.fingerprint.json')
    if os.path.exists(fp):
        meta = json.load(open(fp))
        print(f"  fingerprint: gen_seed={meta.get('gen_seed')} "
              f"mwpm_p_L={meta.get('mwpm_p_L')} flips_sha={meta.get('flips_sha256', '')[:16]}")
    else:
        print(f"  note: no fingerprint at {fp}")

    return dict(n_total=n_total, base_rate_val=base_va, n_det=int(circ.num_detectors))


def check_models(args, fail):
    """Instantiate, smoke-train and checkpoint-reload each architecture."""
    import tensorflow as tf
    from CNNModel import FullRCNNModel
    from StudentModels import build_student, to_sequence

    d, r, p = args.d, args.rounds, args.p
    n_detbits = r * (d ** 2 - 1)
    from eval_on_tail import build_circuit
    n_det = build_circuit(d, p, r).num_detectors

    # Synthetic inputs. The preflight tests plumbing -- shapes, gradients, checkpoint
    # round-trips -- all of which hold for any bits. A falling loss on random labels
    # proves the optimiser is connected, which is check 7's entire claim.
    rng = np.random.default_rng(0)
    n = args.smoke_shots
    det_bits = rng.integers(0, 2, (n, n_detbits), dtype=np.int8)
    det_evts = rng.integers(0, 2, (n, n_det), dtype=np.int8)
    y = rng.integers(0, 2, (n, 1), dtype=np.int8)
    nv = max(n // 5, 2)

    params = {}
    tmpdir = tempfile.mkdtemp(prefix='preflight_d9_')

    for arch in ('rcnn', 'mlp', 'gru'):
        print(f"\n=== model: {arch}  d={d} r={r} ===", flush=True)
        t0 = time.time()

        # ---- 1. instantiate, 4. one batch through, 3. parameter count ---------------
        if arch == 'rcnn':
            def make():
                return FullRCNNModel('ZL', d, KERNEL, r, HIDDEN, npol=NPOL,
                                     stop_round=None, has_nonuniform_response=False,
                                     do_all_data_qubits=False, return_all_rounds=False)
            model = make()
            x, xv = [det_bits, det_evts], [det_bits[:nv], det_evts[:nv]]
            out = model([det_bits[:2], det_evts[:2]])
            loss = 'binary_crossentropy'
        else:
            kw = (dict(student='mlp', hidden=MLP_HIDDEN) if arch == 'mlp'
                  else dict(student='gru', units=GRU_UNITS, hidden=()))

            def make(kw=kw):
                return build_student(d=d, rounds=r, inputs='evts', **kw)
            model = make()
            feats = to_sequence(det_evts, d, r, p) if arch == 'gru' else det_evts
            x, xv = feats, feats[:nv]
            out = model(feats[:2])
            # The students emit a logit. The sigmoid sits outside the model so
            # distillation works in logit space and the FPGA is spared the cost, which
            # the loss has to know about.
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        n_params = int(model.count_params())
        params[arch] = n_params
        print(f"  [1] instantiated in {time.time() - t0:.1f}s")
        print(f"  [3] params {n_params:,}")

        # ---- 2. summary -------------------------------------------------------------
        try:
            model.summary(print_fn=lambda s: print("      " + s))
            print("  [2] summary printed")
        except Exception as e:  # noqa: BLE001 - subclassed models can refuse pre-build
            print(f"  [2] summary unavailable: {type(e).__name__}: {e}")

        # ---- 5. output shape --------------------------------------------------------
        shape = tuple(np.asarray(out).shape)
        print(f"  [4] forward pass ok, output {shape}")
        if shape[0] != 2 or int(np.prod(shape[1:])) != 1:
            fail(f"{arch}: expected one score per shot, got output shape {shape}")
        print("  [5] output shape ok (one score per shot)")

        # ---- 6/7. smoke train, loss must fall ---------------------------------------
        model.compile(optimizer='adam', loss=loss)
        hist = model.fit(x, y, batch_size=args.batch_size, epochs=args.smoke_epochs,
                         validation_data=(xv, y[:nv]), verbose=0)
        losses = hist.history['loss']
        print(f"  [6] smoke trained {args.smoke_epochs} epochs on {n:,} shots "
              f"({time.time() - t0:.0f}s total)")
        print(f"      loss {losses[0]:.5f} -> {losses[-1]:.5f}")
        if not (losses[-1] < losses[0]):
            fail(f"{arch}: loss did not decrease ({losses[0]:.5f} -> {losses[-1]:.5f})")
        print("  [7] loss decreased ok")

        # ---- 8. validation scoring ran ----------------------------------------------
        if 'val_loss' not in hist.history:
            fail(f"{arch}: no val_loss recorded -- validation scoring did not run")
        print(f"  [8] validation scoring ok (val_loss {hist.history['val_loss'][-1]:.5f})")

        # ---- 9. checkpoint reload is bit-identical ----------------------------------
        # Compare predictions: two HDF5 files holding identical arrays differ byte-wise,
        # so hashing the files proves nothing either way.
        wpath = os.path.join(tmpdir, f'{arch}.weights.h5')
        model.save_weights(wpath)
        before = np.asarray(model.predict(xv, batch_size=args.batch_size, verbose=0))
        probe = make()
        _ = probe([det_bits[:2], det_evts[:2]]) if arch == 'rcnn' else probe(xv[:2])
        probe.load_weights(wpath)
        after = np.asarray(probe.predict(xv, batch_size=args.batch_size, verbose=0))
        if not np.array_equal(before, after):
            worst = float(np.max(np.abs(before - after)))
            fail(f"{arch}: reloaded checkpoint predicts differently "
                 f"(max |delta| = {worst:.3e})")
        print("  [9] checkpoint reload bit-identical ok")

    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=9)
    ap.add_argument('--rounds', type=int, default=9)
    ap.add_argument('--p', type=float, default=0.010)
    ap.add_argument('--pool', default=None,
                    help='pool npz to check. Omit to run the model checks only, which '
                         'need no pool and fit on a laptop.')
    ap.add_argument('--n-train', type=int, default=10000000)
    ap.add_argument('--n-val', type=int, default=200000)
    ap.add_argument('--n-test', type=int, default=200000)
    ap.add_argument('--smoke-shots', type=int, default=20000)
    ap.add_argument('--smoke-epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--cpu', action='store_true', help='hide the GPU')
    args = ap.parse_args()

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')
    print(f"[preflight] TF {tf.__version__}  "
          f"GPUs visible: {tf.config.list_physical_devices('GPU')}", flush=True)
    print(f"[preflight] d={args.d} rounds={args.rounds} p={args.p}")

    failures = []

    def fail(msg):
        failures.append(msg)
        print(f"  *** failed: {msg}", flush=True)

    if args.pool:
        check_dataset(args, fail)
    else:
        print("\n=== dataset: skipped (no --pool) ===")

    params = check_models(args, fail)

    print("\n=== parameter count ===")
    for a, n in params.items():
        print(f"  {a:<5} {n:>10,}")
    peak = rss_bytes()
    if peak:
        print(f"\n[preflight] peak RSS {human_bytes(peak)}")

    print()
    if failures:
        print(f"[preflight] {len(failures)} failure(s) -- do not launch:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("[preflight] all checks passed.")


if __name__ == '__main__':
    main()
