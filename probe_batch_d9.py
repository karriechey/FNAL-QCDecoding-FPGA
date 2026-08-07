#!/usr/bin/env python3
# Created: 2026-08-07
# Last updated: 2026-08-07
"""Find the largest RCNN training batch that fits on this GPU at d=9, r=9.

The d=9 RCNN is not broken -- it instantiates and runs a forward pass at the established
configuration, and its parameter count barely moves (69,485 -> 76,117). What grows is
ACTIVATION memory during training: the model evaluates (d-k+1)^2 = 49 kernel positions
across 9 rounds, against 9 positions across 5 rounds at d=5, so the per-batch-element
intermediate tensors grow roughly 10x while the weights do not.

A batch of 10,000 -- the batch every d=5 run used -- was OOM-killed on a 16 GB host. That
says nothing about a 40 GB A100 MIG slice, so this script measures the real ceiling on the
real hardware before anyone changes the experimental design.

Why a subprocess per batch size: TensorFlow does not return device memory to the driver
after an allocation failure, so a batch that OOMs poisons every later measurement in the
same process. Each size therefore gets a clean interpreter, and a size that dies by
SIGKILL (host RAM, uncatchable) is reported as a failure rather than losing the whole run.

Synthetic inputs. Only the tensor shapes drive memory and step time; the bits do not.

  python probe_batch_d9.py                       # ladder 10000 -> 500, d=9 r=9
  python probe_batch_d9.py --d 5 --rounds 5      # the d=5 control, same measurement
  python probe_batch_d9.py --one 10000           # single size, no subprocesses
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

KERNEL = 3
HIDDEN = [100, 100]
NPOL = 2


def measure(d, rounds, batch, steps):
    """Train a few steps at this batch size. Returns (ok, s_per_step, peak_gb, note)."""
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    from CNNModel import FullRCNNModel

    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        # Grow on demand rather than pre-reserving the whole slice, so the peak figure
        # below reflects what the model actually needs.
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass

    n_detbits = rounds * (d ** 2 - 1)
    n = batch * max(steps, 2)
    rng = np.random.default_rng(0)
    det_bits = rng.integers(0, 2, (n, n_detbits), dtype=np.int8)
    det_evts = rng.integers(0, 2, (n, n_detbits), dtype=np.int8)
    y = rng.integers(0, 2, (n, 1), dtype=np.int8)

    model = FullRCNNModel('ZL', d, KERNEL, rounds, HIDDEN, npol=NPOL, stop_round=None,
                          has_nonuniform_response=False, do_all_data_qubits=False,
                          return_all_rounds=False)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    _ = model([det_bits[:1], det_evts[:1]])

    t0 = time.time()
    model.fit([det_bits, det_evts], y, batch_size=batch, epochs=1, verbose=0)
    trace_s = time.time() - t0                       # includes graph tracing
    t0 = time.time()
    model.fit([det_bits, det_evts], y, batch_size=batch, epochs=1, verbose=0)
    steady_s = time.time() - t0
    n_steps = int(np.ceil(n / batch))

    peak_gb = None
    if gpus:
        try:
            peak_gb = tf.config.experimental.get_memory_info('GPU:0')['peak'] / 1024 ** 3
        except Exception:  # noqa: BLE001 - not all builds expose this
            pass
    return steady_s / n_steps, trace_s, peak_gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=9)
    ap.add_argument('--rounds', type=int, default=9)
    ap.add_argument('--batches', type=int, nargs='*',
                    default=[10000, 5000, 2500, 1000, 500])
    ap.add_argument('--steps', type=int, default=4,
                    help='training steps per size; enough to reach steady state')
    ap.add_argument('--one', type=int, default=None,
                    help='measure exactly this batch size in THIS process (used by the '
                         'ladder to isolate each size, and usable directly)')
    args = ap.parse_args()

    if args.one is not None:
        try:
            s_step, trace_s, peak = measure(args.d, args.rounds, args.one, args.steps)
        except Exception as e:  # noqa: BLE001 - a device OOM is a result, not a crash
            print(f"RESULT batch={args.one} ok=0 err={type(e).__name__}", flush=True)
            raise SystemExit(2)
        print(f"RESULT batch={args.one} ok=1 s_per_step={s_step:.4f} "
              f"trace_s={trace_s:.1f} peak_gb={peak if peak else -1:.2f}", flush=True)
        return

    print(f"[probe] RCNN d={args.d} r={args.rounds}, one subprocess per batch size")
    print(f"[probe] {'batch':>7} {'fits':>5} {'s/step':>9} {'GPU peak':>9} "
          f"{'est s/epoch @10M':>17}")
    best = None
    for b in args.batches:
        cmd = [sys.executable, os.path.abspath(__file__), '--d', str(args.d),
               '--rounds', str(args.rounds), '--steps', str(args.steps), '--one', str(b)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in proc.stdout.splitlines() if l.startswith('RESULT')), '')
        if ' ok=1' in line:
            kv = dict(p.split('=', 1) for p in line.split()[1:])
            s_step = float(kv['s_per_step'])
            peak = float(kv['peak_gb'])
            est = s_step * (10000000 / b)
            print(f"[probe] {b:>7} {'yes':>5} {s_step:>9.4f} "
                  f"{(f'{peak:.2f} GB' if peak > 0 else 'n/a'):>9} {est:>17.0f}")
            if best is None:
                best = (b, s_step, est)
        else:
            # -9 is SIGKILL: the host OOM killer, which no exception handler can catch.
            why = 'host OOM (SIGKILL)' if proc.returncode == -9 else 'device OOM / error'
            print(f"[probe] {b:>7} {'NO':>5}   {why}")
    print()
    if best is None:
        print("[probe] nothing fit. The d=9 RCNN cannot train on this device at any "
              "batch size tried -- report before changing the design.")
        raise SystemExit(1)
    b, s_step, est = best
    print(f"[probe] largest batch that fits: {b:,}")
    print(f"[probe] estimated {est / 3600:.1f} h/epoch-equivalent at 10M shots, "
          f"{est * 50 / 3600:.0f} h for 50 epochs, {est * 50 * 3 / 3600:.0f} h for 3 seeds")
    if b == 10000:
        print("[probe] batch 10,000 fits -- keep the d=5 protocol unchanged.")
    else:
        print(f"[probe] batch 10,000 does NOT fit. Using {b:,} at d=9 means the d=5 "
              f"reference should be re-run at {b:,} too, or the distance comparison "
              f"is confounded by the batch change. That is a design decision -- stop "
              f"and raise it.")


if __name__ == '__main__':
    main()
