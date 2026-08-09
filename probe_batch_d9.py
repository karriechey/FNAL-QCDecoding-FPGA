#!/usr/bin/env python3
# Created: 2026-08-07
# Last updated: 2026-08-07
"""Find the largest RCNN training batch that fits at d=9, r=9, and where the limit sits.

At d=9 the RCNN instantiates and runs a forward pass at the established configuration,
and its parameter count barely moves (69,485 -> 76,117). Cost appears once training
starts: the model evaluates (d-k+1)^2 = 49 kernel positions across 9 rounds, against 9
positions across 5 rounds at d=5.

Local measurements on a 16 GB Mac, all killed by SIGKILL during fit():

  batch 10,000, 20,000 shots   killed
  batch  2,000,  4,000 shots   killed
  batch    500,  2,000 shots   killed

Batch-independent, which rules out per-batch activations as the cause and points at host
RAM consumed while tracing the training graph -- 49 kernel positions x 9 unrolled rounds,
forward and backward. A smaller batch would not fix that. The same model's forward pass
runs fine on the same machine.

So this script reports both numbers, device peak and host peak, and the two lead to
different remedies. An EAF pod has far more host RAM than 16 GB, so the d=9 sweep may be
fine there; that is the thing to measure before anyone changes the experimental design.

Why a subprocess per batch size: TensorFlow keeps device memory after an allocation
failure, so a batch that OOMs poisons every later measurement in the same process. Each
size gets a clean interpreter, and a size killed by SIGKILL (host RAM, uncatchable) is
recorded as a failure while the rest of the ladder continues.

Synthetic inputs. Memory and step time follow from the tensor shapes alone.

Run it with the repo's pinned interpreter. The pod's `python` is conda base (TF 2.16 /
Keras 3), which CNNModel.py cannot build under; assert_pinned_stack() stops that early.

  PY=~/QuantumDecoderQKeras/.venv/bin/python
  $PY probe_batch_d9.py                       # ladder 10000 -> 500, d=9 r=9
  $PY probe_batch_d9.py --d 5 --rounds 5      # the d=5 control, same measurement
  $PY probe_batch_d9.py --one 10000           # single size, no subprocesses
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


def assert_pinned_stack():
    """Fail fast unless this interpreter is the pinned TF 2.15 / Keras 2 stack.

    CNNModel.py targets Keras 2. Under Keras 3 it dies during construction, at the
    RCNNStateCorrelator initializer's shape check: `shape != self.shape_exp` compares a
    tuple against a list, so `(4, 18, 9) != [4, 18, 9]` is True on identical values and
    the model never builds. The error names shapes and reads like a model bug, which
    costs a debugging round-trip every time.

    On the EAF pod `python` is /opt/conda/bin/python (TF 2.16 / Keras 3.11). The pinned
    environment is the repo's own .venv, which is what every published run used:

        ~/QuantumDecoderQKeras/.venv/bin/python probe_batch_d9.py
    """
    import tensorflow as tf
    try:
        import keras
        kv = keras.__version__
    except ImportError:
        kv = getattr(tf.keras, '__version__', 'unknown')
    if not str(kv).startswith('2.'):
        raise SystemExit(
            f"[probe] Keras {kv} (TensorFlow {tf.__version__}). CNNModel.py needs "
            f"Keras 2.x -- under Keras 3 it fails in the state-correlator initializer "
            f"on a list-vs-tuple shape comparison, before training starts.\n"
            f"[probe] Use the pinned environment:\n"
            f"[probe]   ~/QuantumDecoderQKeras/.venv/bin/python {sys.argv[0]} ...")
    return tf.__version__, kv


def parse_result(line):
    """Parse a child's RESULT line into a dict.

    `msg=` carries a free-text error and is always last, so it is pulled off first and
    everything before it splits on whitespace. Splitting the whole line naively breaks
    the moment a message contains a space.
    """
    if not line:
        return {}
    body, sep, msg = line.partition(' msg=')
    kv = dict(t.split('=', 1) for t in body.split()[1:] if '=' in t)
    if sep:
        kv['msg'] = msg.strip()
    return kv


def last_rss(stderr):
    """Recover the high-water RSS a killed child printed before it died.

    A SIGKILLed process reports nothing itself, so without this a failed row carries no
    number at all -- and 'died at 15 GB' versus 'died at 9 GB' is what decides whether
    more host RAM fixes the problem.
    """
    vals = [float(l.split()[1]) for l in (stderr or '').splitlines()
            if l.startswith('RSS ') and len(l.split()) > 1]
    return f"  (reached {max(vals):.2f} GB host)" if vals else ""


def rss_gb():
    """Resident set size in GB. Linux reports ru_maxrss in KB, macOS in bytes."""
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (rss * 1024 if sys.platform.startswith('linux') else rss) / 1024 ** 3


def start_rss_ticker(interval=1.0):
    """Print RSS to stderr every `interval` seconds until the process ends.

    A child killed by the OOM killer never reaches its own reporting code, so the parent
    would otherwise learn only that it died -- no number, no way to tell a run that died
    at 15 GB from one that died at 9 GB. That difference decides whether a bigger pod
    solves the problem. stderr survives the kill because the parent captures it, so the
    last ticker line is the recoverable high-water mark.
    """
    import threading

    def tick():
        while True:
            print(f"RSS {rss_gb():.2f}", file=sys.stderr, flush=True)
            time.sleep(interval)

    t = threading.Thread(target=tick, daemon=True)
    t.start()
    return t


def measure(d, rounds, batch, samples, weight_bits=None, forward_only=False):
    """Build the model, then either train a few steps or run one forward pass.

    Returns (s_per_step, trace_s, gpu_peak_gb, host_peak_gb). In forward-only mode
    s_per_step and trace_s describe the single forward call.

    forward_only exists to make the mechanism claim falsifiable. The docstring above
    asserts the forward pass runs where training does not; without a forward-only number
    from this same script that stays an assertion.
    """
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    assert_pinned_stack()
    start_rss_ticker()

    # Quantization inflates the traced graph -- fake-quant nodes at every weight's point
    # of use, plus QDense inside the state decoder -- and graph construction is the host
    # RAM the probe is trying to bound. Exp 10's RCNN arm runs FP32 through train_one.py,
    # so weight_bits=None is the configuration that matches it; the flag exists so a QAT
    # follow-on measures its own cost rather than inheriting this one.
    if weight_bits is None:
        from CNNModel import FullRCNNModel
        build = FullRCNNModel
    else:
        from CNNModel_quantized import build_quantized_rcnn
        import functools
        build = functools.partial(build_quantized_rcnn, weight_bits)

    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        # Grow on demand, so the peak figure below reflects what the model needs.
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass

    n_detbits = rounds * (d ** 2 - 1)
    # Fixed sample count across batch sizes. Sizing n as batch*steps instead makes every
    # larger batch do proportionally more work, so wall-clock rises with batch for a
    # reason that has nothing to do with efficiency -- which is what made 5,000 and
    # 10,000 look like failures when they were simply handed 2x and 4x the shots.
    n = max(samples, batch * 2)
    rng = np.random.default_rng(0)
    det_bits = rng.integers(0, 2, (n, n_detbits), dtype=np.int8)
    det_evts = rng.integers(0, 2, (n, n_detbits), dtype=np.int8)
    y = rng.integers(0, 2, (n, 1), dtype=np.int8)

    model = build('ZL', d, KERNEL, rounds, HIDDEN, npol=NPOL, stop_round=None,
                  has_nonuniform_response=False, do_all_data_qubits=False,
                  return_all_rounds=False)

    if forward_only:
        t0 = time.time()
        _ = model([det_bits[:batch], det_evts[:batch]])
        fwd_s = time.time() - t0
        steady_s, trace_s, n_steps = fwd_s, fwd_s, 1
    else:
        model.compile(optimizer='adam', loss='binary_crossentropy')
        _ = model([det_bits[:1], det_evts[:1]])
        t0 = time.time()
        model.fit([det_bits, det_evts], y, batch_size=batch, epochs=1, verbose=0)
        first_s = time.time() - t0
        t0 = time.time()
        model.fit([det_bits, det_evts], y, batch_size=batch, epochs=1, verbose=0)
        steady_s = time.time() - t0
        # First epoch minus a steady one, so the label means what it says.
        trace_s = first_s - steady_s
        n_steps = int(np.ceil(n / batch))

    peak_gb = None
    if gpus:
        try:
            peak_gb = tf.config.experimental.get_memory_info('GPU:0')['peak'] / 1024 ** 3
        except Exception:  # noqa: BLE001 - not all builds expose this
            pass

    # Host RSS matters as much as device memory here. Three local d=9 runs died by
    # SIGKILL at batch 10,000, 2,000 and 500 alike -- batch-independent, so the cost is
    # in tracing the training graph (49 kernel positions x 9 unrolled rounds, forward and
    # backward) rather than in per-batch activations. A device-memory figure alone would
    # have shown nothing.
    return steady_s / n_steps, trace_s, peak_gb, rss_gb()


def main():
    # Line-buffer stdout. Piping to tee makes it block-buffered, so a ladder that takes
    # tens of minutes per size shows nothing at all until it exits and looks hung.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:      # pragma: no cover - Python < 3.7
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=9)
    ap.add_argument('--rounds', type=int, default=9)
    ap.add_argument('--batches', type=int, nargs='*',
                    default=[10000, 5000, 2500, 1000, 500])
    ap.add_argument('--samples', type=int, default=200000,
                    help='shots per timed epoch, held FIXED across batch sizes so the '
                         'comparison is throughput and not workload. Needs to be large '
                         'enough that per-epoch overhead amortizes: at batch 10,000 this '
                         'default gives 20 steps, against 1,000 in a real 10M epoch.')
    ap.add_argument('--one', type=int, default=None,
                    help='measure exactly this batch size in the current process (the '
                         'ladder uses it to isolate each size; usable directly too)')
    ap.add_argument('--weight-bits', type=int, default=None,
                    help='build through CNNModel_quantized.build_quantized_rcnn at this '
                         'width. Omit for FP32, which is what Exp 10 runs.')
    ap.add_argument('--forward-only', action='store_true',
                    help='build and run one forward pass, skipping training. The control '
                         'for the claim that training is what exhausts memory.')
    ap.add_argument('--stop-on-first', action='store_true',
                    help='stop at the largest batch that fits. The ladder descends, so '
                         'the first success is the answer; omit for the whole curve.')
    ap.add_argument('--timeout', type=int, default=3600,
                    help='per-size seconds. Under swap pressure a d=9 trace thrashes for '
                         'a long time before the OOM killer fires, and a hung ladder is '
                         'indistinguishable from a slow one.')
    args = ap.parse_args()

    if args.one is not None:
        try:
            s_step, trace_s, peak, host = measure(args.d, args.rounds, args.one,
                                                 args.samples, args.weight_bits,
                                                 args.forward_only)
        except Exception as e:  # noqa: BLE001 - record the failure as a result
            # Print the full traceback as well as the machine-readable line. A bare
            # exception name cannot distinguish a genuine device OOM from a plain bug,
            # and the ladder driver only sees this stream.
            import traceback
            traceback.print_exc()
            msg = str(e).replace('\n', ' ')[:200]
            print(f"RESULT batch={args.one} ok=0 err={type(e).__name__} msg={msg}",
                  flush=True)
            raise SystemExit(2)
        print(f"RESULT batch={args.one} ok=1 s_per_step={s_step:.4f} "
              f"trace_s={trace_s:.1f} peak_gb={peak if peak else -1:.2f} "
              f"host_gb={host:.2f}", flush=True)
        return

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    # Version-check in a throwaway subprocess. Importing TensorFlow in the parent would
    # hold it resident for the whole ladder, competing for the exact resource being
    # measured.
    ver = subprocess.run(
        [sys.executable, '-c',
         'import tensorflow as tf;'
         'import keras;'
         'print(tf.__version__, keras.__version__)'],
        capture_output=True, text=True)
    if ver.returncode != 0 or not ver.stdout.strip():
        raise SystemExit(f"[probe] could not import TensorFlow with {sys.executable}\n"
                         f"{ver.stderr.strip()[-400:]}")
    tf_v, keras_v = ver.stdout.split()
    if not keras_v.startswith('2.'):
        raise SystemExit(
            f"[probe] Keras {keras_v} (TensorFlow {tf_v}). CNNModel.py needs Keras 2.x -- "
            f"under Keras 3 it fails in the state-correlator initializer on a "
            f"list-vs-tuple shape comparison, before training starts.\n"
            f"[probe] Use the pinned environment:\n"
            f"[probe]   ~/QuantumDecoderQKeras/.venv/bin/python {sys.argv[0]} ...")
    print(f"[probe] TF {tf_v} / Keras {keras_v}  ({sys.executable})")
    print(f"[probe] RCNN d={args.d} r={args.rounds}, one subprocess per batch size")
    print(f"[probe] {'batch':>7} {'fits':>5} {'s/step':>9} {'gpu peak':>9} "
          f"{'host peak':>10} {'us/sample':>10} {'est s/epoch @10M':>17}")
    print(f"[probe] us/sample is the batch-independent number; compare it across sizes "
          f"and against d=5")
    best = None
    timed_out = []          # sizes that ran out of clock, which says nothing about fit
    for b in args.batches:
        print(f"[probe] {b:>7}  running... (the d=9 training-graph trace is the slow "
              f"part)", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), '--d', str(args.d),
               '--rounds', str(args.rounds), '--samples', str(args.samples),
               '--one', str(b)]
        if args.weight_bits is not None:
            cmd += ['--weight-bits', str(args.weight_bits)]
        if args.forward_only:
            cmd += ['--forward-only']
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=args.timeout)
        except subprocess.TimeoutExpired as e:
            got = last_rss(e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr)
            print(f"[probe] {b:>7} {'t/o':>5}   timeout after {args.timeout}s{got}")
            timed_out.append(b)
            continue
        line = next((l for l in proc.stdout.splitlines() if l.startswith('RESULT')), '')
        if ' ok=1' in line:
            kv = parse_result(line)
            s_step = float(kv['s_per_step'])
            peak = float(kv['peak_gb'])
            host = float(kv['host_gb'])
            per_sample_us = s_step / b * 1e6
            est = per_sample_us * 1e-6 * 10000000
            print(f"[probe] {b:>7} {'yes':>5} {s_step:>9.4f} "
                  f"{(f'{peak:.2f} GB' if peak > 0 else 'n/a'):>9} "
                  f"{f'{host:.2f} GB':>10} {per_sample_us:>10.1f} {est:>17.0f}")
            if best is None:
                best = (b, s_step, est)
            if args.stop_on_first:
                break
        else:
            # -9 is SIGKILL, the host OOM killer, invisible to exception handlers.
            # Anything else is an exception the child caught, and its type has to be
            # shown: a generic label here cannot separate a real device OOM from a bug,
            # and a bug that fails at every batch size looks exactly like a memory wall.
            if proc.returncode == -9:
                why = 'host OOM (SIGKILL)'
            else:
                kv = parse_result(line)
                why = kv.get('err', '')
                if not why:
                    tail = [l for l in proc.stderr.strip().splitlines() if l.strip()]
                    why = tail[-1][:160] if tail else f'exit {proc.returncode}'
                elif kv.get('msg'):
                    why = f"{why}: {kv['msg'][:120]}"
            print(f"[probe] {b:>7} {'no':>5}   {why}{last_rss(proc.stderr)}")
    print()
    if best is None:
        print("[probe] nothing fit. If every size died by SIGKILL, the limit is host "
              "RAM during training-graph construction, which a smaller batch will not "
              "fix -- report before changing the design.")
        raise SystemExit(1)
    b, s_step, est = best
    print(f"[probe] largest batch that fits: {b:,}")
    if args.forward_only:
        # No training happened, so a training-cost projection here would be a fabricated
        # number wearing a real number's units.
        print(f"[probe] forward-only: {s_step:.2f}s for one forward pass of {b:,} "
              f"shots. No training cost measured -- the host peak in the row above is "
              f"the build plus forward pass only.")
        return
    # Exp 10 passes an explicit validation block (--val-start/--val-n), which disables
    # validation_split, so an epoch really does train on all 10M shots and this figure is
    # not inflated by a split. EarlyStopping(patience=5) is the only reason a run comes
    # in under 50 epochs.
    print(f"[probe] {est / 3600:.1f} h/epoch at 10M shots, "
          f"{est * 50 / 3600:.0f} h for 50 epochs, "
          f"{est * 50 * 3 / 3600:.0f} h for 3 seeds")
    print("[probe] an epoch trains on all 10M (Exp 10 uses an explicit validation block, "
          "so validation_split is off); early stopping is the only thing that shortens it")
    # A timeout is a statement about the clock, not about capacity. Treating one as
    # "does not fit" would recommend a protocol change on evidence that does not support
    # it, so any size that timed out is reported as unresolved.
    if timed_out:
        print(f"[probe] unresolved (timeout, not a memory limit): "
              f"{', '.join(f'{t:,}' for t in timed_out)}")
        print(f"[probe] these hit --timeout {args.timeout}s while still running. Re-run "
              f"with a larger --timeout before concluding anything about batch size.")
        if any(t > b for t in timed_out):
            print(f"[probe] {max(timed_out):,} may well fit -- {b:,} is only the largest "
                  f"size that finished in time.")
            return
    if 10000 not in args.batches:
        print("[probe] batch 10,000 was not in this run's --batches, so this says nothing "
              "about the protocol batch. Re-run without --batches for that.")
    elif b == 10000:
        print("[probe] batch 10,000 fits -- keep the d=5 protocol unchanged.")
    else:
        print(f"[probe] batch 10,000 exceeds this device. Using {b:,} at d=9 means the d=5 "
              f"reference should be re-run at {b:,} too, so the distance comparison "
              f"stays controlled. That is a design decision -- stop and raise it.")


if __name__ == '__main__':
    main()
