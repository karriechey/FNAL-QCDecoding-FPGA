#!/usr/bin/env python3
# Created: 2026-08-06
# Last updated: 2026-08-06
"""Shared slice guard and environment guard for the d=5, r=5 parameter-reduction study.

Why this file exists
--------------------
The study defines four disjoint index ranges over the formal pool:

    train        [        0, 10,000,000)   gradient updates
    val_select   [20,000,000, 20,100,000)  checkpoint selection, early stopping, any tuning
    val_report   [20,100,000, 20,200,000)  the number that gets reported
    sealed test  [20,200,000, 20,400,000)  never read anywhere in this study

Section 2.3 of the specification requires that the sealed range is never indexed, passed
to a model, decoded, summarised, or used for any decision, and that the check lives in ONE
place rather than being copied into each script, because per-script copies drift apart.

Every script in the study therefore routes each index range it is about to use through
`assert_slice_allowed(start, stop, purpose)`. The call raises on any overlap with the
sealed block and appends a line to a per-process log so the pilot can confirm afterwards
that every range really did pass through the guard and that none was rejected silently.

Loading the whole NPZ into host memory is not counted as test access -- the pool is a
single file and there is no way to read the training prefix without mapping the array.
What is prohibited is any semantic use of, or slice-level access to, the sealed indices.

Environment guard
-----------------
`assert_deterministic_env()` must be called BEFORE TensorFlow is imported. It checks that
the launcher exported TF_DETERMINISTIC_OPS=1 and TF_CUDNN_DETERMINISTIC=1, because both
are read by TensorFlow at import time and setting them afterwards is silently ineffective.
`assert_tf_version()` is called after the import and hard-fails on anything that is not
TensorFlow 2.15.x. Both fail loudly rather than warning and continuing.

Determinism here reduces uncontrolled implementation-level variation so that same-seed
reruns are comparable. It is not a claim of bitwise-identical reproducibility; that would
have to be demonstrated by a smoke test on the target EAF stack before being asserted.
"""
import json
import os
import sys
import time

# ---------------------------------------------------------------------------------------
# The single definition of the study's partitions. Every script imports these rather than
# writing the literals again, so a change here cannot leave one script on old boundaries.
# ---------------------------------------------------------------------------------------
TRAIN_START, TRAIN_STOP = 0, 10_000_000
VAL_SELECT_START, VAL_SELECT_STOP = 20_000_000, 20_100_000
VAL_REPORT_START, VAL_REPORT_STOP = 20_100_000, 20_200_000
SEALED_START, SEALED_STOP = 20_200_000, 20_400_000

# ---------------------------------------------------------------------------------------
# Smoke-test override. SLICE_GUARD_SMOKE_DIVISOR divides every boundary above by an
# integer, which preserves the partition geometry (four disjoint blocks in the same order,
# the same 100:1 train-to-report ratio) on a pool small enough to build and train on in
# minutes. It exists so the smoke test can exercise the REAL code paths rather than a
# mock; it is not a way to shrink a formal run.
#
# Three things keep it from leaking into a formal result:
#   1. it prints a banner on every process that uses it;
#   2. is_smoke() is recorded in every artifact these scripts write;
#   3. the frozen manifest records the boundaries it was built with, and the collation
#      compares every run's recorded slice against the manifest's, so a smoke run and a
#      formal manifest can never be mixed.
# The production launcher unsets the variable explicitly.
# ---------------------------------------------------------------------------------------
_SMOKE_DIVISOR = int(os.environ.get('SLICE_GUARD_SMOKE_DIVISOR', '1'))
if _SMOKE_DIVISOR != 1:
    if _SMOKE_DIVISOR < 1:
        raise SystemExit('[slice-guard] SLICE_GUARD_SMOKE_DIVISOR must be >= 1')
    for _name in ('TRAIN_STOP', 'VAL_SELECT_START', 'VAL_SELECT_STOP',
                  'VAL_REPORT_START', 'VAL_REPORT_STOP', 'SEALED_START', 'SEALED_STOP'):
        globals()[_name] = globals()[_name] // _SMOKE_DIVISOR
    print('=' * 78)
    print(f"[slice-guard] SMOKE MODE: every partition boundary divided by "
          f"{_SMOKE_DIVISOR}. Results from this process are NOT study results.")
    print('=' * 78, flush=True)

POOL_MIN_SHOTS = SEALED_STOP  # the pool must contain at least this many shots


def is_smoke():
    """True when the partition boundaries have been scaled down for a smoke test."""
    return _SMOKE_DIVISOR != 1


def smoke_divisor():
    return _SMOKE_DIVISOR

# Named ranges, for scripts that want to refer to a partition by name in a log line.
PARTITIONS = {
    'train': (TRAIN_START, TRAIN_STOP),
    'val_select': (VAL_SELECT_START, VAL_SELECT_STOP),
    'val_report': (VAL_REPORT_START, VAL_REPORT_STOP),
    'sealed_test': (SEALED_START, SEALED_STOP),
}

# Every call to assert_slice_allowed appends a record here, and to the file named by
# SLICE_GUARD_LOG if that variable is set. The pilot checks this log.
_GUARD_CALLS = []
_LOG_PATH = os.environ.get('SLICE_GUARD_LOG')


class SealedTestAccess(RuntimeError):
    """Raised when a requested index range touches the sealed test partition."""


def assert_slice_allowed(start, stop, purpose):
    """Approve one half-open index range [start, stop) for one stated purpose.

    Arguments:
      start, stop: half-open global pool indices, exactly as they will be used to slice.
      purpose:     free text naming what the range is for, e.g. 'train prefix',
                   'val_select validation_data', 'val_report scoring', 'MWPM baseline'.
                   It is recorded in the log so a reviewer can see what each range was for.

    Returns the (start, stop) pair so a caller can write
        lo, hi = assert_slice_allowed(...)
    and be unable to use an unchecked range by accident.

    Raises SealedTestAccess on any overlap with [SEALED_START, SEALED_STOP), and
    ValueError on a malformed or out-of-pool range.
    """
    start, stop = int(start), int(stop)
    if stop <= start:
        raise ValueError(f"empty or reversed range [{start}, {stop}) for {purpose!r}")
    if start < 0:
        raise ValueError(f"negative start {start} for {purpose!r}")
    if stop > POOL_MIN_SHOTS:
        raise ValueError(
            f"range [{start}, {stop}) for {purpose!r} runs past the end of the declared "
            f"pool ({POOL_MIN_SHOTS:,} shots)")

    # Half-open overlap test: two ranges intersect when each starts before the other ends.
    overlaps_sealed = start < SEALED_STOP and SEALED_START < stop
    record = dict(t=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  start=start, stop=stop, n=stop - start, purpose=str(purpose),
                  script=os.path.basename(sys.argv[0]) or 'interactive',
                  allowed=not overlaps_sealed)
    _GUARD_CALLS.append(record)
    line = (f"[slice-guard] {'ALLOW' if not overlaps_sealed else 'REJECT'} "
            f"[{start:,}, {stop:,})  n={stop - start:,}  purpose={purpose}")
    print(line, flush=True)
    if _LOG_PATH:
        with open(_LOG_PATH, 'a') as fh:
            fh.write(json.dumps(record) + '\n')

    if overlaps_sealed:
        raise SealedTestAccess(
            f"range [{start:,}, {stop:,}) requested for {purpose!r} overlaps the sealed "
            f"test partition [{SEALED_START:,}, {SEALED_STOP:,}). Section 2.3 forbids "
            f"reading it for any purpose in this study.")
    return start, stop


def guard_calls():
    """Every range this process passed through the guard, in call order."""
    return list(_GUARD_CALLS)


def guard_summary():
    """One printable block listing every guarded range. Called at the end of each run."""
    lines = [f"[slice-guard] {len(_GUARD_CALLS)} range(s) requested this process:"]
    for c in _GUARD_CALLS:
        lines.append(f"[slice-guard]   {'ALLOW' if c['allowed'] else 'REJECT'} "
                     f"[{c['start']:,}, {c['stop']:,})  {c['purpose']}")
    n_rejected = sum(1 for c in _GUARD_CALLS if not c['allowed'])
    lines.append(f"[slice-guard] rejected: {n_rejected} (a rejection also raised, so a "
                 f"run that reached this line with a rejection would be a guard bug)")
    return '\n'.join(lines)


def assert_partitions_disjoint():
    """Confirm the four declared partitions do not overlap each other. Cheap, run always."""
    names = list(PARTITIONS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            a0, a1 = PARTITIONS[a]
            b0, b1 = PARTITIONS[b]
            if a0 < b1 and b0 < a1:
                raise AssertionError(
                    f"partitions {a} [{a0:,}, {a1:,}) and {b} [{b0:,}, {b1:,}) overlap")
    print("[slice-guard] the four declared partitions are mutually disjoint", flush=True)
    return True


# ---------------------------------------------------------------------------------------
# Environment guards
# ---------------------------------------------------------------------------------------
REQUIRED_ENV = {'TF_DETERMINISTIC_OPS': '1', 'TF_CUDNN_DETERMINISTIC': '1'}


def assert_deterministic_env():
    """Fail unless the launcher exported the determinism variables BEFORE this process.

    Must be called before `import tensorflow`. TensorFlow reads both variables during its
    own import, so a script that sets them afterwards gets no determinism and no warning.
    This function refuses to set them itself for exactly that reason: if they are missing,
    the process was started wrong and cannot be repaired in place.
    """
    if 'tensorflow' in sys.modules:
        raise AssertionError(
            "assert_deterministic_env() was called after TensorFlow was already imported. "
            "The determinism variables are read at import time, so this ordering gives no "
            "determinism. Move the call above the import.")
    missing = [k for k, v in REQUIRED_ENV.items() if os.environ.get(k) != v]
    if missing:
        raise SystemExit(
            "[env] MISSING required determinism variables: "
            + ', '.join(f"{k}={REQUIRED_ENV[k]}" for k in missing)
            + ". Export them in the launcher, before the Python process starts:\n"
              "    export TF_DETERMINISTIC_OPS=1\n"
              "    export TF_CUDNN_DETERMINISTIC=1")
    print("[env] TF_DETERMINISTIC_OPS=1 TF_CUDNN_DETERMINISTIC=1 present before import",
          flush=True)
    return True


def assert_tf_version(tf_module):
    """Hard-fail on anything that is not TensorFlow 2.15.x. Call right after the import."""
    v = tf_module.__version__
    if not v.startswith('2.15.'):
        raise SystemExit(f"[env] TensorFlow {v} -- this study is pinned to 2.15.x. STOP.")
    print(f"[env] TensorFlow {v} (pinned 2.15.x)", flush=True)
    return v


if __name__ == '__main__':
    # Running the module directly is a self-test of the guard, used by the smoke test.
    assert_partitions_disjoint()
    assert_slice_allowed(TRAIN_START, TRAIN_STOP, 'self-test: train prefix')
    assert_slice_allowed(VAL_SELECT_START, VAL_SELECT_STOP, 'self-test: val_select')
    assert_slice_allowed(VAL_REPORT_START, VAL_REPORT_STOP, 'self-test: val_report')
    try:
        assert_slice_allowed(SEALED_START, SEALED_STOP, 'self-test: sealed (must raise)')
    except SealedTestAccess as e:
        print(f"[slice-guard] correctly refused the sealed range: {e}")
    else:
        raise SystemExit("[slice-guard] FAILED: the sealed range was not refused")
    # A range that only clips the sealed block by one shot must also be refused.
    try:
        assert_slice_allowed(VAL_REPORT_START, SEALED_START + 1,
                             'self-test: one-shot overlap (must raise)')
    except SealedTestAccess:
        print("[slice-guard] correctly refused a one-shot overlap")
    else:
        raise SystemExit("[slice-guard] FAILED: a one-shot overlap was not refused")
    print(guard_summary())
