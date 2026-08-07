#!/usr/bin/env python3
# Created: 2026-08-06
# Last updated: 2026-08-06
"""Collate the d=5, r=5 parameter-reduction sweep. Covers specification section 7.3.

Reads every per-run summary and per-shot report under --out-root, checks each one against
the frozen manifest, and emits:

    param_reduction_r5_summary_<stamp>.csv   one row per architecture x fraction
    param_reduction_r5_runs_<stamp>.csv      one row per run, tidy, for plotting
    param_reduction_r5_mcnemar_<stamp>.csv   paired tests, one row per comparison
    param_reduction_r5_problems_<stamp>.csv  missing, duplicate, incomplete, mismatched

Paired tests, all on val_report and all from the saved per-shot vectors:
  * each run against the RECOMPUTED MWPM baseline on the same slice;
  * each reduced rung against the 100% rung of the same architecture and the same seed.

The three seeds share val_report, so their p-values are not independent and are NOT
Fisher-combined. They are reported separately, and this script refuses to produce a
combined p-value at all.

Trains nothing and reads no sealed shot: everything here comes from files whose own
metadata records the val_report slice, and that slice is re-checked against the guard.
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys
import time

import numpy as np

from slice_guard_r5 import (assert_slice_allowed, VAL_REPORT_START, VAL_REPORT_STOP,
                            guard_summary)
from eval_on_tail import mcnemar_from_correct

LOG = '[collate]'
EXPECTED_SEEDS = (0, 1, 2)


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_runs(out_root, manifest_hash, problems):
    """Every per-run summary under out_root, validated. Malformed runs go to `problems`."""
    runs = []
    for path in sorted(glob.glob(os.path.join(out_root, '**', 'valreport_*.json'),
                                 recursive=True)):
        try:
            d = json.load(open(path))
        except Exception as e:
            problems.append(dict(kind='unreadable_summary', path=path, detail=repr(e)))
            continue
        required = ('architecture', 'target_fraction', 'actual_params', 'seed', 'p_L',
                    'val_report_slice', 'manifest_sha256', 'per_shot_file', 'git_sha')
        missing = [k for k in required if k not in d]
        if missing:
            problems.append(dict(kind='incomplete_summary', path=path,
                                 detail=f'missing keys: {missing}'))
            continue
        if d['manifest_sha256'] != manifest_hash:
            problems.append(dict(
                kind='manifest_hash_mismatch', path=path,
                detail=f"run recorded {d['manifest_sha256'][:16]}..., frozen manifest is "
                       f"{manifest_hash[:16]}... -- this run describes a different set of "
                       f"configurations and is excluded"))
            continue
        if list(d['val_report_slice']) != [VAL_REPORT_START, VAL_REPORT_STOP]:
            problems.append(dict(kind='wrong_report_slice', path=path,
                                 detail=f"scored {d['val_report_slice']}, expected "
                                        f"[{VAL_REPORT_START}, {VAL_REPORT_STOP})"))
            continue
        # The summary records the per-shot file by absolute path, which is an EAF path
        # when the run was trained there. After a tarball transfer that path does not
        # exist locally, so fall back to the sibling file with the matching name before
        # calling the run incomplete.
        if not os.path.exists(d['per_shot_file']):
            sibling = os.path.join(os.path.dirname(path),
                                   'pershot_' + os.path.basename(path).replace('.json',
                                                                               '.npz'))
            if os.path.exists(sibling):
                d['per_shot_file'] = sibling
            else:
                problems.append(dict(
                    kind='missing_per_shot', path=path,
                    detail=f"recorded {d['per_shot_file']} does not exist and neither "
                           f"does the sibling {sibling}"))
                continue
        d['_summary_path'] = path
        runs.append(d)
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-root', required=True, help='the sweep output root')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--mwpm', default=None,
                    help='mwpm_val_report_r5.json from pool_integrity_r5.py. Required: '
                         'the baseline for this study is the RECOMPUTED value on '
                         'val_report, never a number carried in from another slice.')
    ap.add_argument('--mwpm-per-shot', default=None,
                    help='mwpm_val_report_per_shot.npz, for the paired tests')
    ap.add_argument('--collate-dir', default=None,
                    help='where the output CSVs go (default: --out-root)')
    args = ap.parse_args()

    collate_dir = args.collate_dir or args.out_root
    os.makedirs(collate_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())

    manifest = json.load(open(args.manifest))
    mhash = sha256_file(args.manifest)
    print(f"{LOG} manifest {args.manifest}  sha256={mhash}")
    # The manifest records the partition boundaries it was built with. If they differ from
    # the ones this process is using, one of the two is a smoke-scaled run and mixing them
    # would produce a nonsense table.
    if list(manifest['partitions']['val_report']) != [VAL_REPORT_START, VAL_REPORT_STOP]:
        raise SystemExit(
            f"{LOG} the manifest was built with val_report="
            f"{manifest['partitions']['val_report']} but this process is using "
            f"[{VAL_REPORT_START}, {VAL_REPORT_STOP}). One of them is smoke-scaled "
            f"(SLICE_GUARD_SMOKE_DIVISOR). STOP.")

    # The MWPM baseline. Section 2.2: recomputed on val_report, never hard-coded.
    if args.mwpm is None:
        raise SystemExit(f"{LOG} --mwpm is required. Run pool_integrity_r5.py first; the "
                         f"study's baseline is the value it measures on val_report.")
    mwpm = json.load(open(args.mwpm))
    if list(mwpm['slice']) != [VAL_REPORT_START, VAL_REPORT_STOP]:
        raise SystemExit(f"{LOG} the MWPM baseline was computed on {mwpm['slice']}, not on "
                         f"val_report [{VAL_REPORT_START}, {VAL_REPORT_STOP}). STOP.")
    mwpm_pL = float(mwpm['p_L'])
    print(f"{LOG} recomputed MWPM on val_report: p_L={mwpm_pL:.6f} "
          f"({mwpm['residual_errors']:,}/{mwpm['n']:,})")

    mwpm_shots = None
    if args.mwpm_per_shot:
        z = np.load(args.mwpm_per_shot, allow_pickle=True)
        assert_slice_allowed(int(z['slice_start']), int(z['slice_stop']),
                             'MWPM per-shot vectors for pairing')
        mwpm_shots = dict(shot_idx=z['shot_idx'], correct=z['mwpm_correct'].astype(bool))

    problems = []
    runs = load_runs(args.out_root, mhash, problems)
    print(f"{LOG} {len(runs)} valid run summaries, {len(problems)} problem(s) so far")

    # --- expected vs found --------------------------------------------------------------
    expected = [(c['architecture'], c['target_fraction'], int(c['actual_params']))
                for c in manifest['configurations'] if c['reachable'] == 'YES']
    for c in manifest['configurations']:
        if c['reachable'] != 'YES':
            problems.append(dict(
                kind='rung_dropped_by_design', path='',
                detail=f"{c['architecture']} @ {c['target_fraction']}: {c['reason']}"))

    index = {}
    for r in runs:
        key = (r['architecture'], float(r['target_fraction']), int(r['actual_params']),
               int(r['seed']))
        if key in index:
            problems.append(dict(kind='duplicate_run', path=r['_summary_path'],
                                 detail=f"{key} already seen at "
                                        f"{index[key]['_summary_path']}"))
            continue
        index[key] = r

    for arch, frac, params in expected:
        for seed in EXPECTED_SEEDS:
            if (arch, frac, params, seed) not in index:
                problems.append(dict(
                    kind='missing_run', path='',
                    detail=f"{arch} fraction={frac} params={params} seed={seed}"))

    # Never let two different measured parameter counts hide under one fraction label.
    by_label = {}
    for (arch, frac, params, seed) in index:
        by_label.setdefault((arch, frac), set()).add(params)
    for (arch, frac), counts in by_label.items():
        if len(counts) > 1:
            problems.append(dict(
                kind='parameter_count_collision', path='',
                detail=f"{arch} @ {frac} has runs at {sorted(counts)} parameters; these "
                       f"are different models and are reported as separate rows, never "
                       f"averaged together"))

    # --- per-run rows and paired tests ---------------------------------------------------
    run_rows, mcnemar_rows = [], []
    per_shot_cache = {}

    def per_shot(r):
        p = r['per_shot_file']
        if p not in per_shot_cache:
            z = np.load(p, allow_pickle=True)
            assert_slice_allowed(int(z['slice_start']), int(z['slice_stop']),
                                 f"per-shot vectors for {os.path.basename(p)}")
            per_shot_cache[p] = dict(shot_idx=z['shot_idx'],
                                     correct=z['correctness'].astype(bool),
                                     pred=z['predicted_class'],
                                     prob=z['predicted_probability'])
        return per_shot_cache[p]

    for key in sorted(index):
        arch, frac, params, seed = key
        r = index[key]
        run_rows.append(dict(
            architecture=arch, target_fraction=frac, actual_params=params, seed=seed,
            n_train=r.get('n_train', ''), p_L=r['p_L'],
            residual_errors=r.get('residual_errors', ''),
            mwpm_p_L=mwpm_pL, xMWPM=r['p_L'] / mwpm_pL,
            predicted_positive_rate=r.get('predicted_positive_rate', ''),
            prob_std=r.get('prob_std', ''),
            checkpoint=r.get('checkpoint', ''), git_sha=r.get('git_sha', ''),
            manifest_sha256=r['manifest_sha256'],
            val_report_slice=f"[{VAL_REPORT_START}, {VAL_REPORT_STOP})",
            summary_path=r['_summary_path']))

        if mwpm_shots is not None:
            ps = per_shot(r)
            if not np.array_equal(ps['shot_idx'], mwpm_shots['shot_idx']):
                problems.append(dict(kind='shot_index_mismatch_vs_mwpm',
                                     path=r['per_shot_file'],
                                     detail='the run and the MWPM baseline do not cover '
                                            'the same global shot indices; no paired test '
                                            'is computed for this run'))
            else:
                mc = mcnemar_from_correct(ps['correct'], mwpm_shots['correct'])
                mcnemar_rows.append(dict(
                    comparison='model_vs_MWPM', architecture=arch, target_fraction=frac,
                    actual_params=params, seed=seed,
                    b_model_only_correct=mc['rcnn_only'], c_other_only_correct=mc['mwpm_only'],
                    n_discordant=mc['n_discordant'], both_right=mc['both_right'],
                    both_wrong=mc['both_wrong'], net_model_wins=mc['net_rcnn_wins'],
                    chi2_cc=mc['mcnemar_chi2_cc'], p_chi2=mc['p_chi2'],
                    p_exact=mc['p_exact'],
                    note='paired on val_report; seeds share this slice so these p-values '
                         'are not independent and are not combined'))

    # rung vs its own architecture's 100% rung, same seed
    hundred = {(a, p, s): r for (a, f, p, s), r in index.items() if abs(f - 1.0) < 1e-9}
    hundred_by_arch_seed = {}
    for (a, f, p, s), r in index.items():
        if abs(f - 1.0) < 1e-9:
            hundred_by_arch_seed[(a, s)] = r
    for key in sorted(index):
        arch, frac, params, seed = key
        if abs(frac - 1.0) < 1e-9:
            continue
        ref = hundred_by_arch_seed.get((arch, seed))
        if ref is None:
            problems.append(dict(
                kind='missing_100pct_reference', path=index[key]['_summary_path'],
                detail=f"{arch} seed={seed} has no 100% run, so the rung-vs-100% paired "
                       f"test cannot be computed"))
            continue
        a_ps, b_ps = per_shot(index[key]), per_shot(ref)
        if not np.array_equal(a_ps['shot_idx'], b_ps['shot_idx']):
            problems.append(dict(kind='shot_index_mismatch_vs_100pct',
                                 path=index[key]['per_shot_file'],
                                 detail='shot indices differ from the 100% run'))
            continue
        mc = mcnemar_from_correct(a_ps['correct'], b_ps['correct'])
        mcnemar_rows.append(dict(
            comparison='rung_vs_100pct_same_seed', architecture=arch,
            target_fraction=frac, actual_params=params, seed=seed,
            b_model_only_correct=mc['rcnn_only'], c_other_only_correct=mc['mwpm_only'],
            n_discordant=mc['n_discordant'], both_right=mc['both_right'],
            both_wrong=mc['both_wrong'], net_model_wins=mc['net_rcnn_wins'],
            chi2_cc=mc['mcnemar_chi2_cc'], p_chi2=mc['p_chi2'], p_exact=mc['p_exact'],
            note=f"b = this rung correct where the 100% rung was wrong; "
                 f"c = the reverse. Reference: {os.path.basename(ref['_summary_path'])}"))

    # --- per architecture x fraction summary ---------------------------------------------
    summary_rows = []
    groups = {}
    for (arch, frac, params, seed), r in index.items():
        groups.setdefault((arch, frac, params), []).append(r)
    for (arch, frac, params) in sorted(groups):
        rs = groups[(arch, frac, params)]
        pls = [float(r['p_L']) for r in rs]
        n = len(pls)
        summary_rows.append(dict(
            architecture=arch, target_fraction=frac, actual_params=params,
            n_seeds=n, seeds=' '.join(str(int(r['seed'])) for r in sorted(rs, key=lambda x: x['seed'])),
            mean_p_L=statistics.fmean(pls),
            sd_p_L_ddof1=(statistics.stdev(pls) if n > 1 else ''),
            min_p_L=min(pls), max_p_L=max(pls),
            mwpm_p_L=mwpm_pL, mean_xMWPM=statistics.fmean(pls) / mwpm_pL,
            complete=('YES' if n == len(EXPECTED_SEEDS) else f'NO ({n}/{len(EXPECTED_SEEDS)})')))

    # --- write ---------------------------------------------------------------------------
    def write_csv(name, rows, cols=None):
        path = os.path.join(collate_dir, name)
        if not rows:
            with open(path, 'w') as fh:
                fh.write('# no rows\n')
            return path
        cols = cols or list(rows[0])
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return path

    p1 = write_csv(f'param_reduction_r5_summary_{stamp}.csv', summary_rows)
    p2 = write_csv(f'param_reduction_r5_runs_{stamp}.csv', run_rows)
    p3 = write_csv(f'param_reduction_r5_mcnemar_{stamp}.csv', mcnemar_rows)
    p4 = write_csv(f'param_reduction_r5_problems_{stamp}.csv', problems,
                   cols=['kind', 'path', 'detail'])

    print()
    print(f"{LOG} per architecture x fraction (p_L on val_report, MWPM = {mwpm_pL:.6f}):")
    for r in summary_rows:
        sd = r['sd_p_L_ddof1']
        sd_s = f"{sd:.6f}" if sd != '' else '   n/a'
        print(f"{LOG}   {r['architecture']:5s} {r['target_fraction']:>5.0%} "
              f"{r['actual_params']:7,d} params  n={r['n_seeds']}  "
              f"mean={r['mean_p_L']:.6f}  sd={sd_s}  "
              f"min={r['min_p_L']:.6f}  max={r['max_p_L']:.6f}  "
              f"xMWPM={r['mean_xMWPM']:.3f}  complete={r['complete']}")
    print()
    if problems:
        print(f"{LOG} {len(problems)} problem(s):")
        for p in problems:
            print(f"{LOG}   {p['kind']}: {p['detail']}")
    else:
        print(f"{LOG} no missing, duplicate, incomplete or mismatched runs")
    print()
    print(f"{LOG} p-values above are per seed. The three seeds share val_report, so they "
          f"are not independent; this script does not Fisher-combine them and no combined "
          f"p-value should be quoted.")
    for p in (p1, p2, p3, p4):
        print(f"{LOG} wrote {p}")
    print(guard_summary())
    return 1 if any(p['kind'] not in ('rung_dropped_by_design',) for p in problems) else 0


if __name__ == '__main__':
    sys.exit(main())
