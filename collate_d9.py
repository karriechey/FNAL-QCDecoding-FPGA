#!/usr/bin/env python3
# Created: 2026-08-07
# Last updated: 2026-08-07
"""Exp 10 collation: the d=5,r=5 vs d=9,r=9 distance-scaling tables.

Reads the per-run outputs the three launchers leave behind and emits the two tables the
study is defined to produce -- one row per seed, then one row per architecture per
distance -- as both CSV and Markdown.

Where each number comes from, and why they are not read the same way:

  RCNN     $OUTROOT/val_scores_best_ckpt.csv, written by eval_on_tail.py, NOT the
           per-run training CSV. A run's own p_L is whatever weights sat in memory when
           fit() returned, which is the restored best only if early stopping fired -- so
           it is not comparable across seeds. The re-scored .best checkpoint is.
  MLP/GRU  the per-run CSVs train_student.py writes, one file per run, which already
           carry the .best-selected p_L along with alpha, n_params and the pool hashes.

MWPM is never taken from a model run's mwpm_p_L column. That column is suppressed under
an explicit --pool anyway, and the one time it was not, a stale value (0.0451 against the
true 0.049405 on the same shots) propagated into a sweep. The baseline comes from the
pool's own mwpm_baseline.csv / fingerprint, matched on (d, p, rounds).

  python collate_d9.py                        # both distances, default EAF paths
  python collate_d9.py --rt /path/to/pulled   # a downloaded tarball tree
  python collate_d9.py --out-dir results_exp10
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

# The d=5, r=5, 10M reference. Hard-label students, matching the regime the d=9 runs use;
# every value is scored on that pool's validation block [20.0M, 20.2M).
D5_REFERENCE = dict(
    pool='data_d5_p0.010_r5_FORMAL.npz',
    partition='validation [20,000,000, 20,200,000)',
    mwpm_p_L=0.084215,
    params=dict(rcnn=69485, mlp=69389, gru=69441),
)


def read_mwpm(rt, d, p, rounds, pool_dir):
    """MWPM p_L for this exact (d, p, rounds), from the pool build -- never from a run."""
    for cand in (os.path.join(rt, pool_dir, 'mwpm_baseline.csv'),):
        if not os.path.exists(cand):
            continue
        for row in csv.DictReader(open(cand)):
            if (int(row['d']) == d and abs(float(row['p']) - p) < 1e-9
                    and int(row['rounds']) == rounds):
                return float(row['mwpm_p_L']), cand
    # Fall back to the pool fingerprint, which records the same decode.
    for fp in glob.glob(os.path.join(rt, pool_dir, f'data_d{d}_p{p:.3f}_r{rounds}_*.fingerprint.json')):
        meta = json.load(open(fp))
        if meta.get('mwpm_p_L') is not None:
            return float(meta['mwpm_p_L']), fp
    return None, None


def read_rcnn(outroot):
    """Per-seed rows from the re-scored .best checkpoints."""
    path = os.path.join(outroot, 'val_scores_best_ckpt.csv')
    rows = []
    if not os.path.exists(path):
        return rows, path
    for r in csv.DictReader(open(path)):
        m = re.search(r'seed(\d+)', r['weights'])
        n = re.search(r'ntr(\d+)', r['weights'])
        rows.append(dict(architecture='rcnn', seed=int(m.group(1)) if m else -1,
                         n_train=int(n.group(1)) if n else 0,
                         p_L=float(r['p_L']), params=None,
                         epochs=None, train_time_s=None))
    return rows, path


def read_students(outdir):
    """Per-seed rows from train_student.py's per-run CSVs."""
    rows = []
    for f in sorted(glob.glob(os.path.join(outdir, 'd9hard_*.csv'))
                    + sorted(glob.glob(os.path.join(outdir, 'r5hard_*.csv')))):
        if '.superseded_' in os.path.basename(f):
            continue
        for r in csv.DictReader(open(f)):
            # Guard the regime explicitly: a distilled row in a hard-label directory
            # would silently corrupt the distance comparison, which is exactly the
            # failure the study's controls section calls out.
            if abs(float(r.get('alpha', 1.0)) - 1.0) > 1e-9:
                print(f"  SKIP {os.path.basename(f)}: alpha={r['alpha']}, not hard label")
                continue
            rows.append(dict(architecture=r['student'], seed=int(r['seed']),
                             n_train=int(r['n_train']), p_L=float(r['p_L']),
                             params=int(r['n_params']),
                             epochs=int(r.get('epochs_ran') or 0),
                             train_time_s=float(r.get('train_time_s') or 0)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rt', default=os.path.expanduser('~/rcnn_threshold'))
    ap.add_argument('--rcnn-out', default=None, help='default $RT/out_d9_rcnn')
    ap.add_argument('--student-out', default=None, help='default $RT/out_d9_student_hard')
    ap.add_argument('--n-train', type=int, default=10000000)
    ap.add_argument('--out-dir', default='results_exp10')
    args = ap.parse_args()

    rcnn_out = args.rcnn_out or os.path.join(args.rt, 'out_d9_rcnn')
    stud_out = args.student_out or os.path.join(args.rt, 'out_d9_student_hard')

    mwpm9, mwpm9_src = read_mwpm(args.rt, 9, 0.010, 9, 'pools_d9')
    if mwpm9 is None:
        raise SystemExit(f"[collate] no d=9 MWPM baseline under {args.rt}/pools_d9 -- "
                         f"the pool build writes it. Cannot compute xMWPM. STOP.")
    print(f"[collate] d=9 MWPM {mwpm9:.6f}  <- {mwpm9_src}")
    print(f"[collate] d=5 MWPM {D5_REFERENCE['mwpm_p_L']:.6f}  <- {D5_REFERENCE['pool']}")

    rcnn_rows, rcnn_src = read_rcnn(rcnn_out)
    print(f"[collate] rcnn {len(rcnn_rows)} rows <- {rcnn_src}")
    stud_rows = read_students(stud_out)
    print(f"[collate] students {len(stud_rows)} rows <- {stud_out}")

    rows = []
    for r in rcnn_rows + stud_rows:
        if r['n_train'] != args.n_train:
            continue
        r.update(distance=9, rounds=9, mwpm=mwpm9, xmwpm=r['p_L'] / mwpm9)
        rows.append(r)
    if not rows:
        print("[collate] no d=9 runs found yet -- reporting the d=5 reference only")

    os.makedirs(args.out_dir, exist_ok=True)
    per_seed = os.path.join(args.out_dir, 'exp10_per_seed.csv')
    fields = ['distance', 'rounds', 'architecture', 'seed', 'params', 'n_train',
              'p_L', 'xMWPM', 'epochs', 'train_time_s']
    with open(per_seed, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r['architecture'], r['seed'])):
            w.writerow(dict(distance=9, rounds=9, architecture=r['architecture'],
                            seed=r['seed'], params=r['params'] or '',
                            n_train=r['n_train'], p_L=round(r['p_L'], 6),
                            xMWPM=round(r['xmwpm'], 4), epochs=r['epochs'] or '',
                            train_time_s=r['train_time_s'] or ''))
    print(f"[collate] per-seed -> {per_seed}")

    # Architecture-level aggregate, both distances side by side.
    g = defaultdict(list)
    for r in rows:
        g[r['architecture']].append(r)

    lines = ["| distance | architecture | params | mean p_L | min p_L | max p_L | "
             "mean xMWPM | seeds |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    agg_path = os.path.join(args.out_dir, 'exp10_by_architecture.csv')
    with open(agg_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['distance', 'rounds', 'architecture', 'params', 'n_seeds',
                    'mean_p_L', 'std_p_L', 'min_p_L', 'max_p_L', 'mean_xMWPM', 'mwpm_p_L'])
        for arch in ('rcnn', 'mlp', 'gru'):
            # d=5 reference row. p_L is filled from the existing study only if the
            # collation is pointed at a tree that holds it; otherwise the parameter count
            # alone carries the scaling comparison, and the p_L cell says so.
            w.writerow([5, 5, arch, D5_REFERENCE['params'][arch], '', '', '', '', '',
                        '', D5_REFERENCE['mwpm_p_L']])
            lines.append(f"| 5 | {arch} | {D5_REFERENCE['params'][arch]:,} | "
                         f"see d=5 study | | | | |")
            v = g.get(arch, [])
            if not v:
                continue
            pls = [x['p_L'] for x in v]
            xs = [x['xmwpm'] for x in v]
            params = next((x['params'] for x in v if x['params']), '')
            w.writerow([9, 9, arch, params, len(v), round(st.mean(pls), 6),
                        round(st.stdev(pls), 6) if len(pls) > 1 else 0.0,
                        round(min(pls), 6), round(max(pls), 6),
                        round(st.mean(xs), 4), mwpm9])
            lines.append(f"| 9 | {arch} | {params:,} | {st.mean(pls):.5f} | "
                         f"{min(pls):.5f} | {max(pls):.5f} | {st.mean(xs):.3f} | {len(v)} |")
    print(f"[collate] aggregate -> {agg_path}")

    md = os.path.join(args.out_dir, 'exp10_summary.md')
    with open(md, 'w') as fh:
        fh.write("# Exp 10 — distance scaling, d=r=5 vs d=r=9, 10M shots, p=0.010\n\n")
        fh.write(f"MWPM d=9,r=9: **{mwpm9:.6f}**  ({mwpm9_src})\n\n")
        fh.write(f"MWPM d=5,r=5: **{D5_REFERENCE['mwpm_p_L']:.6f}**  "
                 f"({D5_REFERENCE['partition']})\n\n")
        fh.write("Hard-label students (alpha=1.0) at both distances. "
                 "All values on the validation partition; both sealed test blocks unread.\n\n")
        fh.write("\n".join(lines) + "\n")
    print(f"[collate] markdown -> {md}")
    print("\n" + "\n".join(lines))


if __name__ == '__main__':
    main()
