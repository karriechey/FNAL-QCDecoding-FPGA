#!/usr/bin/env python3
# Created: 2026-08-06
# Last updated: 2026-08-06
"""Find the width settings that hit the parameter-budget rungs of the d=5, r=5 study.

This script trains nothing. It instantiates and builds each candidate model, reads
`model.count_params()` off the built model, and writes a frozen manifest describing the
exact constructor arguments for every architecture x budget rung.

Why measured counts and not a formula
-------------------------------------
The parameter count is nonlinear in the width knob for every architecture here:

  MLP  hidden=(h, h)   params = h^2 + 123h + 1     (quadratic in h)
  GRU  units=u         params = 3u^2 + 76u + 1     (quadratic in u)
  RCNN hidden_specs=[h, h]
                       params = 56,684 + h^2 + 28h + 1
                                ^^^^^^ fixed non-decoder stack, independent of h

Scaling a width by 0.5 therefore does not scale the count by 0.5, and for the RCNN the
count does not go to zero at all as the width shrinks. Nothing in this file estimates a
count from a width ratio; every number printed comes from a built model.

The RCNN parameter floor
------------------------
`FullRCNNModel.__init__` exposes exactly one width knob, `hidden_specs`, which is handed
to `StateDecoder` -- the decoder head MLP. Every other component (DetectorBitStateEmbedder,
DetectorEventStateEmbedder, the CNNKernel instances, the state correlators and the kernel
combiners) is sized by code_distance, kernel_distance, rounds and npol, none of which may
be changed here: kernel_distance and npol change the decoding mechanism itself and would
break comparability with the rest of the project. The RCNN therefore cannot be shrunk
below the size of its non-decoder stack, and any rung under that floor is unreachable and
is dropped rather than faked.

Usage
-----
    export TF_DETERMINISTIC_OPS=1 TF_CUDNN_DETERMINISTIC=1
    python parameter_search_r5.py --out-dir rcnn_threshold/param_reduction_r5

Writes, into --out-dir:
    param_manifest_r5.json    the frozen manifest, consumed by the launcher
    param_manifest_r5.csv     the same rows, for reading and plotting
    rcnn_component_breakdown_r5.csv   per-component RCNN parameter accounting

The manifest is frozen once written: the launcher recomputes its SHA-256 and aborts if it
does not match the hash recorded at freeze time. Rerunning with --force overwrites it and
mints a new hash, which invalidates every run tagged with the old one.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

# The environment guard must run before TensorFlow is imported, because TensorFlow reads
# the determinism variables during its own import.
from slice_guard_r5 import (assert_deterministic_env, assert_tf_version, is_smoke,
                            smoke_divisor, TRAIN_START, TRAIN_STOP,
                            VAL_SELECT_START, VAL_SELECT_STOP,
                            VAL_REPORT_START, VAL_REPORT_STOP,
                            SEALED_START, SEALED_STOP)

LOG = '[param-search]'

# The four budget rungs, as fractions of each architecture's own 100% configuration.
DEFAULT_FRACTIONS = (1.0, 0.5, 0.25, 0.10)
TOLERANCE = 0.05  # +/-5%, per specification section 4

# The 100% configurations, which are the matched-capacity settings the r=5 ladder already
# used. They are re-measured here, never assumed; the expected values are only compared
# against, and a mismatch stops the script.
EXPECTED_100PCT = {'rcnn': 69485, 'gru': 69441, 'mlp': 69389}

# Fixed circuit setting for the whole study.
D, ROUNDS, KERNEL, NPOL, P = 5, 5, 3, 2, 0.010
HIDDEN_LAYERS = 2  # both the RCNN decoder head and the MLP student keep two hidden layers


def git_sha():
    """The commit the manifest was produced at, recorded so a rung can be traced back."""
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'],
                                        stderr=subprocess.DEVNULL).decode().strip()
        return sha + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------
# Model builders. Each returns a BUILT model, so count_params() is meaningful.
# ---------------------------------------------------------------------------------------
def build_rcnn(hidden, np_mod):
    """FullRCNNModel with a two-layer decoder head of width `hidden`.

    Constructor arguments are copied from train_one.py so the searched model is the model
    that will be trained: obs_type 'ZL', kernel_distance 3, npol 2, no stop_round, uniform
    response, not all data qubits, final round only.
    """
    from CNNModel import FullRCNNModel
    m = FullRCNNModel('ZL', D, KERNEL, ROUNDS, [hidden] * HIDDEN_LAYERS,
                      npol=NPOL, stop_round=None, has_nonuniform_response=False,
                      do_all_data_qubits=False, return_all_rounds=False)
    # One forward pass on a two-shot dummy batch builds every weight. The values are
    # irrelevant -- only the shapes are being measured -- so no pool read is needed.
    det_bits = np_mod.zeros((2, ROUNDS * (D ** 2 - 1)), dtype=np_mod.float32)
    det_evts = np_mod.zeros((2, 120), dtype=np_mod.float32)
    _ = m([det_bits, det_evts])
    return m


def build_gru(units):
    """GRU student: one recurrent layer of width `units`, straight into the logit head.

    `hidden=()` is the configuration the r=5 student ladder actually launched, via the
    bare `--hidden` flag in eaf_run_student_ladder_r5.sh (argparse nargs='*' with no
    values gives an empty list). build_student's default of (128, 128) would add a dense
    head and give 103,989 parameters at units=140, not the 69,441 on record. reset_after
    stays pinned to False inside build_gru_student.
    """
    from StudentModels import build_student
    return build_student('gru', d=D, rounds=ROUNDS, inputs='evts', units=units, hidden=())


def build_mlp(hidden):
    """MLP student: two hidden layers of equal width `hidden`, then the logit head."""
    from StudentModels import build_student
    return build_student('mlp', d=D, rounds=ROUNDS, inputs='evts',
                         hidden=(hidden,) * HIDDEN_LAYERS)


BUILDERS = {'gru': build_gru, 'mlp': build_mlp}   # rcnn needs numpy, handled separately


# ---------------------------------------------------------------------------------------
# RCNN component accounting
# ---------------------------------------------------------------------------------------
def rcnn_component_breakdown(model, np_mod):
    """Group the RCNN's unique weights into named components and total each.

    Grouping is by variable identity, not by name: the CNN kernels reuse variable names
    across kernel offsets (three variables are all called
    DetectorBitStateEmbedder_npol2_params_nondiag), so a name-keyed dictionary would
    silently collapse them and undercount. Identity keying also means a genuinely shared
    weight is counted once, which is what section 5 requires.
    """
    unique = {}
    for v in model.variables:
        unique.setdefault(id(v), v)

    def component_of(name):
        # Names are matched against the component that owns the weight. Only the first
        # entry is sized by hidden_specs; everything below it is the fixed stack whose
        # total is the floor.
        if 'state_decoder' in name:
            return 'decoder_head (StateDecoder, sized by hidden_specs)'
        if 'DetectorBitStateEmbedder' in name:
            return 'DetectorBitStateEmbedder'
        if 'DetectorEventStateEmbedder' in name:
            return 'DetectorEventStateEmbedder'
        if 'CNNKernel' in name and 'w_det_bits' in name:
            return 'CNNKernel detector-bit weights'
        if 'CNNKernel' in name and 'w_det_evts' in name:
            return 'CNNKernel detector-event weights'
        if 'CNNStateCorrelator' in name:
            return 'CNNStateCorrelator (state evolution + bias)'
        if 'TripletStateProbEmbedder' in name:
            return 'TripletStateProbEmbedder'
        if 'RCNNLeadInKernel' in name:
            return 'RCNNLeadInKernel triplet-state weights'
        if 'RCNNRecurrenceKernel' in name:
            return 'RCNNRecurrenceKernel triplet-state weights'
        if 'Translation' in name:
            return 'kernel combiners (TranslationFrac / TranslationPhase)'
        # Anything unrecognised is reported under its own name rather than swept into a
        # catch-all, so a component added upstream shows up instead of hiding.
        return f'UNCLASSIFIED ({name})'

    groups = {}
    for v in unique.values():
        name = getattr(v, 'name', '?')
        key = component_of(name)
        g = groups.setdefault(key, dict(component=key, n_vars=0, n_params=0, shapes=[]))
        g['n_vars'] += 1
        g['n_params'] += int(np_mod.prod(v.shape))
        g['shapes'].append(f"{name}{tuple(v.shape)}")

    total_unique = sum(g['n_params'] for g in groups.values())
    reported = int(model.count_params())
    if total_unique != reported:
        raise SystemExit(
            f"{LOG} component accounting does not close: unique variables sum to "
            f"{total_unique:,} but count_params() reports {reported:,}. Investigate "
            f"before using any number from this script.")
    return groups, total_unique


def rcnn_floor(np_mod):
    """The RCNN's hard parameter floor: everything that is not the decoder head.

    Measured, not derived: build at two different head widths and subtract the head's own
    analytic size, then confirm both give the same non-decoder total. If they disagree,
    something outside the head is also responding to hidden_specs and the floor claim
    would be wrong.
    """
    floors = {}
    for h in (100, 50):
        m = build_rcnn(h, np_mod)
        groups, total = rcnn_component_breakdown(m, np_mod)
        head = sum(g['n_params'] for k, g in groups.items() if k.startswith('decoder_head'))
        floors[h] = total - head
    vals = set(floors.values())
    if len(vals) != 1:
        raise SystemExit(f"{LOG} the non-decoder stack changed with hidden_specs: {floors}. "
                         f"The floor claim in section 5 does not hold. STOP.")
    return vals.pop()


# ---------------------------------------------------------------------------------------
# Width search
# ---------------------------------------------------------------------------------------
def search_width(builder, target, lo, hi, extra_kw=None):
    """Return the integer width whose measured count is closest to `target`.

    Brute force over [lo, hi]: every candidate is built and measured, because the count is
    quadratic in the width and a bisection on a formula is exactly the shortcut this study
    forbids. The range is small (a few hundred builds, seconds each on CPU) so there is no
    reason to be clever.
    """
    best = None
    for w in range(lo, hi + 1):
        m = builder(w) if extra_kw is None else builder(w, **extra_kw)
        n = int(m.count_params())
        err = abs(n - target)
        if best is None or err < best[2]:
            best = (w, n, err)
        # The counts increase monotonically in the width, so once past the target by more
        # than the best error seen, nothing further can improve.
        if n > target and best[2] < n - target:
            break
    return best[0], best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', default='rcnn_threshold/param_reduction_r5',
                    help='where the manifest, CSV and breakdown are written')
    ap.add_argument('--fractions', type=float, nargs='+', default=list(DEFAULT_FRACTIONS),
                    help='budget rungs as fractions of each 100%% configuration')
    ap.add_argument('--force', action='store_true',
                    help='overwrite an existing manifest. This mints a new manifest hash '
                         'and invalidates every run already tagged with the old one.')
    ap.add_argument('--cpu', action='store_true', default=True,
                    help='hide GPUs (default: on -- this script only builds models)')
    args = ap.parse_args()

    assert_deterministic_env()          # must precede the TensorFlow import
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import numpy as np
    import tensorflow as tf
    assert_tf_version(tf)
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, 'param_manifest_r5.json')
    if os.path.exists(manifest_path) and not args.force:
        raise SystemExit(f"{LOG} {manifest_path} exists and is frozen. Pass --force only "
                         f"if you intend to invalidate every run tagged with its hash.")

    sha = git_sha()
    print(f"{LOG} git {sha}   d={D} p={P} rounds={ROUNDS} kernel={KERNEL} npol={NPOL}")
    print(f"{LOG} this script: {os.path.abspath(__file__)}")
    print()

    # --- 1. Re-measure the three 100% configurations ------------------------------------
    print(f"{LOG} re-measuring the 100% configurations (expected values are compared, "
          f"never assumed)")
    ref = {}
    ref['rcnn'] = (dict(hidden=100, hidden_layers=HIDDEN_LAYERS, npol=NPOL,
                        kernel=KERNEL),
                   int(build_rcnn(100, np).count_params()))
    ref['gru'] = (dict(units=140, hidden=[]), int(build_gru(140).count_params()))
    ref['mlp'] = (dict(hidden=[209, 209]), int(build_mlp(209).count_params()))
    bad = []
    for arch, (cfg, n) in ref.items():
        exp = EXPECTED_100PCT[arch]
        ok = (n == exp)
        print(f"{LOG}   {arch:5s} {cfg}  measured={n:,}  expected={exp:,}  "
              f"{'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            bad.append((arch, n, exp))
    if bad:
        raise SystemExit(
            f"{LOG} the current matched-capacity definitions do not reproduce the "
            f"reference counts: {bad}. Section 4 says stop and report. STOP.")
    print()

    # --- 2. RCNN component breakdown and floor ------------------------------------------
    print(f"{LOG} RCNN component breakdown at hidden_specs=[100, 100]")
    m100 = build_rcnn(100, np)
    groups, total = rcnn_component_breakdown(m100, np)
    floor = rcnn_floor(np)
    for k in sorted(groups, key=lambda k: -groups[k]['n_params']):
        g = groups[k]
        print(f"{LOG}   {g['n_params']:8,d}  ({g['n_vars']:3d} vars)  {k}")
    print(f"{LOG}   {'-' * 60}")
    print(f"{LOG}   {total:8,d}  sum of UNIQUE variables == count_params() -- accounting closes")
    print(f"{LOG}   non-decoder floor = {floor:,} parameters "
          f"({100.0 * floor / total:.1f}% of the 100% model)")
    print(f"{LOG}   the smallest legal RCNN (hidden_specs=[1, 1]) is "
          f"{int(build_rcnn(1, np).count_params()):,}")
    print()

    breakdown_path = os.path.join(args.out_dir, 'rcnn_component_breakdown_r5.csv')
    with open(breakdown_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['component', 'n_unique_variables', 'n_params', 'variable_shapes'])
        for k in sorted(groups, key=lambda k: -groups[k]['n_params']):
            g = groups[k]
            w.writerow([k, g['n_vars'], g['n_params'], ' | '.join(sorted(g['shapes']))])
        w.writerow(['TOTAL (unique) == count_params()', len(list(m100.variables)), total, ''])
        w.writerow(['NON-DECODER FLOOR', '', floor,
                    'hidden_specs cannot reduce this; kernel_distance and npol are fixed '
                    'by comparability with the rest of the project'])
    print(f"{LOG} wrote {breakdown_path}")
    print()

    # --- 3. Width search for every architecture x rung ----------------------------------
    rows = []
    for arch in ('rcnn', 'gru', 'mlp'):
        base = ref[arch][1]
        for frac in args.fractions:
            target = int(round(base * frac))
            if arch == 'rcnn':
                if frac == 1.0:
                    width, actual = 100, base
                elif target < floor:
                    rows.append(dict(
                        architecture=arch, target_fraction=frac, target_params=target,
                        actual_params='', relative_error='', width='',
                        constructor_args='', reachable='NO',
                        reason=f'target {target:,} is below the RCNN non-decoder floor of '
                               f'{floor:,}; reachable only by changing kernel_distance or '
                               f'npol, which section 5 forbids'))
                    print(f"{LOG} rcnn {frac:>5.0%}  target={target:8,d}  UNREACHABLE "
                          f"(floor={floor:,})")
                    continue
                else:
                    width, actual = search_width(lambda h: build_rcnn(h, np), target, 1, 400)
            elif arch == 'gru':
                width, actual = (140, base) if frac == 1.0 else search_width(build_gru, target, 1, 200)
            else:
                width, actual = (209, base) if frac == 1.0 else search_width(build_mlp, target, 1, 400)

            rel_err = (actual - target) / target
            if arch == 'rcnn':
                cargs = dict(hidden=width, hidden_layers=HIDDEN_LAYERS, npol=NPOL,
                             kernel=KERNEL)
            elif arch == 'gru':
                cargs = dict(units=width, hidden=[])
            else:
                cargs = dict(hidden=[width] * HIDDEN_LAYERS)
            flag = 'OK' if abs(rel_err) <= TOLERANCE else 'OUT OF TOLERANCE'
            rows.append(dict(architecture=arch, target_fraction=frac,
                             target_params=target, actual_params=actual,
                             relative_error=round(rel_err, 6), width=width,
                             constructor_args=json.dumps(cargs), reachable='YES',
                             reason=''))
            print(f"{LOG} {arch:5s} {frac:>5.0%}  target={target:8,d}  "
                  f"actual={actual:8,d}  rel_err={rel_err:+.2%}  {cargs}  {flag}")
    print()

    # --- 4. Freeze the manifest ---------------------------------------------------------
    csv_path = os.path.join(args.out_dir, 'param_manifest_r5.csv')
    cols = ['architecture', 'target_fraction', 'target_params', 'actual_params',
            'relative_error', 'width', 'constructor_args', 'reachable', 'reason']
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    manifest = dict(
        study='parameter_reduction_r5',
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        git_sha=sha,
        produced_by=os.path.basename(__file__),
        tensorflow=tf.__version__,
        circuit=dict(d=D, p=P, rounds=ROUNDS, kernel_distance=KERNEL, npol=NPOL),
        # Taken from slice_guard_r5, not written out as literals, so a manifest built
        # under the smoke divisor records the boundaries it was actually built with and
        # can never be mistaken for a formal one.
        partitions=dict(train=[TRAIN_START, TRAIN_STOP],
                        val_select=[VAL_SELECT_START, VAL_SELECT_STOP],
                        val_report=[VAL_REPORT_START, VAL_REPORT_STOP],
                        sealed_test_DO_NOT_READ=[SEALED_START, SEALED_STOP]),
        smoke=is_smoke(), smoke_divisor=smoke_divisor(),
        reference_100pct={a: dict(config=c, measured_params=n)
                          for a, (c, n) in ref.items()},
        rcnn_non_decoder_floor=floor,
        rcnn_component_breakdown={k: g['n_params'] for k, g in groups.items()},
        tolerance=TOLERANCE,
        configurations=rows,
    )
    with open(manifest_path, 'w') as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
    mhash = sha256_file(manifest_path)
    # The hash is recorded in a sidecar rather than inside the manifest, since a hash
    # written into the file it describes cannot be self-consistent.
    with open(manifest_path + '.sha256', 'w') as fh:
        fh.write(f"{mhash}  {os.path.basename(manifest_path)}\n")

    print(f"{LOG} wrote {csv_path}")
    print(f"{LOG} wrote {manifest_path}")
    print(f"{LOG} MANIFEST SHA-256 = {mhash}")
    print(f"{LOG} the manifest is now frozen. The launcher recomputes this hash and "
          f"aborts on a mismatch.")

    n_runs = sum(1 for r in rows if r['reachable'] == 'YES') * 3
    print(f"{LOG} reachable configurations: "
          f"{sum(1 for r in rows if r['reachable'] == 'YES')} x 3 seeds = {n_runs} runs")
    return 0


if __name__ == '__main__':
    sys.exit(main())
