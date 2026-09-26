#!/usr/bin/env python3
# Created: 2026-08-17
# Last modified: 2026-08-17
"""Collate the d=5, r=5, p=0.004 GRU parameter-reduction and distillation study.

Reads the per-run CSVs written by train_student.py under a sweep's runs/ directory, groups
them by (parameter budget, supervision arm), and emits one tidy CSV per point plus a
seed-level long CSV. The plotting script consumes the tidy CSV and nothing else, so the
figure cannot silently disagree with the table.

Reported per point:
  n_seeds, mean p_L, sample standard deviation (ddof=1), min, max, and the ratio to MWPM
  on the evaluation block.

Also reported, per parameter budget, the distilled-minus-hard difference paired by seed.
Pairing matters: the two arms at a given width share seeds, initialisation stream and
shuffle order, so the per-seed difference removes the seed-to-seed scatter that dominates
a comparison of two means at n=3. With three seeds this is a description of the observed
spread, not a significance test, and it is reported as such.

What is deliberately NOT here: a paired McNemar test against MWPM or between arms.
train_student.py records aggregate p_L, not per-shot predictions, so shot-level pairing is
not reconstructible from these files. Adding it means re-inferencing every checkpoint on
the evaluation block, which is a separate step.

  python collate_gru_compression_p004.py \
      --runs ~/rcnn_threshold/results/gru_compression_r5_p004_<stamp>/runs \
      --out  ~/rcnn_threshold/results/gru_compression_r5_p004_<stamp> \
      --mwpm 0.007572
"""
import argparse
import csv
import glob
import os
import statistics as st
from collections import defaultdict

# The four widths this study runs, and the parameter count each one actually builds to,
# measured with count_params() on a built model. Held here as a table rather than
# recomputed so collation never has to import TensorFlow; verify() cross-checks it against
# what the runs themselves recorded.
UNITS_TO_PARAMS = {140: 69441, 70: 20021, 46: 9845, 34: 6053}

# What the study expects to find. A missing run is a hole in an error bar, so it is named
# rather than quietly averaged over.
EXPECTED_SEEDS = (0, 1, 2)


def gru_params(units):
    """Trainable parameters of the d=5, r=5 evts GRU student at a given width.

    3 gates x (input 24 + recurrent U + bias 1) x U, plus the U+1 linear head. Matches
    count_params() on the built model for every width in UNITS_TO_PARAMS; kept as a
    formula so an unlisted width still collates.
    """
    return 3 * units * (24 + units + 1) + units + 1


def parse_tag(tag):
    """Pull (role, units, mode, seed) back out of a run tag.

    Tags look like gruc_p004_teacher_u140_hard_seed0_ntr10000000 or
    gruc_p004_student_u46_distill_seed2_ntr10000000. Parsed positionally from the end so a
    changed prefix does not break it.
    """
    parts = tag.split('_')
    try:
        role = 'teacher' if 'teacher' in parts else 'student'
        units = int(next(p for p in parts if p.startswith('u') and p[1:].isdigit())[1:])
        mode = 'distill' if 'distill' in parts else 'hard'
        seed = int(next(p for p in parts if p.startswith('seed'))[4:])
    except (StopIteration, ValueError):
        return None
    return role, units, mode, seed


def load_runs(runs_dir):
    """Every completed run under runs/, one dict per row, with the tag fields resolved."""
    rows = []
    for path in sorted(glob.glob(os.path.join(runs_dir, 'gruc_p004_*.csv'))):
        # train_student.py archives a re-run's predecessor as <tag>.superseded_<utc>.csv
        # rather than overwriting it. Those are history, not results.
        if '.superseded_' in os.path.basename(path):
            continue
        tag = os.path.basename(path)[:-len('.csv')]
        parsed = parse_tag(tag)
        if parsed is None:
            print(f"  [warn] unparseable tag, skipped: {tag}")
            continue
        role, units, mode, seed = parsed
        for r in csv.DictReader(open(path)):
            r['_tag'] = tag
            r['_role'] = role
            r['_units'] = units
            r['_mode'] = mode
            r['_seed'] = seed
            # The run's own count_params() is authoritative; the formula is the fallback
            # for an older CSV that predates the n_params column, and verify() checks the
            # two against each other.
            recorded = r.get('n_params')
            r['_params'] = int(recorded) if recorded not in (None, '') else gru_params(units)
            rows.append(r)
    return rows


def verify(rows):
    """Report holes and inconsistencies rather than averaging over them."""
    problems = []

    by_tag = defaultdict(int)
    for r in rows:
        by_tag[r['_tag']] += 1
    for tag, n in sorted(by_tag.items()):
        if n > 1:
            problems.append(f"duplicate rows for {tag}: {n} rows in one CSV")

    # A run's own record of its width must agree with what its filename claims.
    for r in rows:
        recorded = r.get('units') or r.get('gru_units')
        if recorded not in (None, '') and int(recorded) != r['_units']:
            problems.append(f"{r['_tag']}: tag says units={r['_units']}, "
                            f"the run recorded units={recorded}")
        if r['_params'] != gru_params(r['_units']):
            problems.append(f"{r['_tag']}: run recorded {r['_params']:,} params, the "
                            f"d=5 r=5 GRU formula gives {gru_params(r['_units']):,} at "
                            f"units={r['_units']} -- different architecture or layout")
        if r['_units'] in UNITS_TO_PARAMS and r['_params'] != UNITS_TO_PARAMS[r['_units']]:
            problems.append(f"{r['_tag']}: {r['_params']:,} params, the study's table "
                            f"says {UNITS_TO_PARAMS[r['_units']]:,}")

    # Every run in a comparable set must share the recipe, or the comparison is not one.
    for field in ('batch_size', 'n_train', 'epochs'):
        seen = {r.get(field) for r in rows if r.get(field) not in (None, '')}
        if len(seen) > 1:
            problems.append(f"runs disagree on {field}: {sorted(seen)}")

    present = {(r['_units'], r['_mode'], r['_seed']) for r in rows}
    units_seen = sorted({u for u, _, _ in present}, reverse=True)
    for u in units_seen:
        for mode in ('hard', 'distill'):
            have = {s for uu, mm, s in present if uu == u and mm == mode}
            if not have:
                continue  # that arm was not requested at this width (e.g. the teacher)
            missing = [s for s in EXPECTED_SEEDS if s not in have]
            if missing:
                problems.append(f"units={u} {mode}: missing seeds {missing}")
    return problems


def summarize(rows, mwpm):
    """One record per (units, mode): the point that goes on the plot."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r['_units'], r['_mode'], r['_role'])].append(r)

    out = []
    for (units, mode, role), rs in groups.items():
        pls = sorted(float(r['p_L']) for r in rs)
        # Wall-clock and epochs actually run. Both belong beside p_L: a point that stopped
        # at the epoch cap is not the same measurement as one that converged, and the cost
        # of a configuration is part of what a compression study is reporting.
        times = [float(r['train_time_s']) for r in rs if r.get('train_time_s')]
        eps = [int(r['epochs_ran']) for r in rs if r.get('epochs_ran')]
        cap = [int(r['epochs']) for r in rs if r.get('epochs')]
        out.append({
            'role': role,
            'units': units,
            'params': rs[0]['_params'],
            'mode': mode,
            'n_seeds': len(pls),
            'seeds': ' '.join(str(r['_seed']) for r in sorted(rs, key=lambda r: r['_seed'])),
            'mean_p_L': round(st.mean(pls), 6),
            'std_p_L': round(st.stdev(pls), 6) if len(pls) > 1 else '',
            'min_p_L': round(min(pls), 6),
            'max_p_L': round(max(pls), 6),
            'mwpm_p_L': mwpm,
            'ratio_vs_mwpm': round(st.mean(pls) / mwpm, 4),
            'mean_train_time_s': round(st.mean(times), 1) if times else '',
            'min_train_time_s': round(min(times), 1) if times else '',
            'max_train_time_s': round(max(times), 1) if times else '',
            'mean_epochs_ran': round(st.mean(eps), 1) if eps else '',
            'epoch_cap': cap[0] if cap else '',
            # How many of this point's seeds stopped because they ran out of epochs rather
            # than because validation loss stopped improving. A point with cap hits is
            # measured at a fixed budget, not at convergence.
            'n_hit_cap': sum(1 for e, c in zip(eps, cap) if e >= c) if eps and cap else '',
            'mean_s_per_epoch': round(st.mean(times) / st.mean(eps), 2) if times and eps else '',
        })
    out.sort(key=lambda r: (-r['params'], r['mode']))
    return out


def paired_differences(rows):
    """Distilled minus hard at each width, paired by seed.

    Returns one record per width that has both arms on at least one shared seed. Negative
    mean_delta means distillation gave the lower logical error rate.
    """
    by = {(r['_units'], r['_mode'], r['_seed']): float(r['p_L']) for r in rows}
    widths = sorted({u for u, _, _ in by}, reverse=True)
    out = []
    for u in widths:
        deltas = []
        for s in EXPECTED_SEEDS:
            if (u, 'hard', s) in by and (u, 'distill', s) in by:
                deltas.append(by[(u, 'distill', s)] - by[(u, 'hard', s)])
        if not deltas:
            continue
        out.append({
            'units': u,
            'params': gru_params(u),
            'n_paired_seeds': len(deltas),
            'mean_delta': round(st.mean(deltas), 6),
            'std_delta': round(st.stdev(deltas), 6) if len(deltas) > 1 else '',
            'deltas': ' '.join(f"{d:+.6f}" for d in deltas),
        })
    return out


def audit_teacher_selection(path):
    """Re-derive the teacher choice from its own record, and report what it was made on.

    The launcher selects the distillation teacher on best_val_loss over the validation
    block. This re-checks that after the fact: the selected seed must be the argmin of
    best_val_loss, and must NOT have been chosen because it had the lowest evaluation p_L.
    Both are reported, because "the two happen to agree" and "the choice was made on the
    evaluation block" look identical in a single number and are not the same thing.

    Returns a list of problem strings, empty if the selection is clean.
    """
    if not os.path.exists(path):
        return [f"no teacher_selection.csv at {path} -- cannot audit the teacher choice"]

    rows = list(csv.DictReader(open(path)))
    if not rows:
        return [f"{os.path.basename(path)} is empty"]

    selected = [r for r in rows if r['selected'] == '1']
    problems = []
    if len(selected) != 1:
        return [f"{len(selected)} rows marked selected in {os.path.basename(path)}; "
                "exactly one teacher may be chosen"]

    chosen = selected[0]
    by_val = min(rows, key=lambda r: float(r['best_val_loss']))
    by_eval = min(rows, key=lambda r: float(r['eval_p_L']))

    if chosen['seed'] != by_val['seed']:
        problems.append(
            f"teacher seed {chosen['seed']} is NOT the lowest best_val_loss "
            f"(that is seed {by_val['seed']}) -- the selection did not follow the protocol")

    agree = ' (which is also the lowest evaluation p_L -- coincidence, not the criterion)' \
        if chosen['seed'] == by_eval['seed'] else \
        f" (the lowest evaluation p_L was seed {by_eval['seed']}, which was NOT chosen -- " \
        "confirming the criterion was the validation block)"
    print(f"[collate] teacher = seed {chosen['seed']}, best_val_loss="
          f"{float(chosen['best_val_loss']):.6f}{agree}")
    return problems


def write_csv(path, records, fields):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True, help='the sweep runs/ directory')
    ap.add_argument('--out', required=True, help='directory for the collated CSVs')
    ap.add_argument('--mwpm', type=float, required=True,
                    help='MWPM p_L on the exact evaluation block [15.2M, 17.0M). '
                         '0.007572 for the d=5 p=0.004 19M pool (exp13 table of record). '
                         'Never taken from a per-run CSV column: train_one.lookup_mwpm() '
                         'keys only on (d, p, rounds) and does not know which shots were '
                         'evaluated.')
    args = ap.parse_args()

    rows = load_runs(args.runs)
    if not rows:
        raise SystemExit(f"[collate] no runs found under {args.runs}")
    print(f"[collate] {len(rows)} runs from {args.runs}")

    problems = verify(rows)
    problems += audit_teacher_selection(
        os.path.join(args.out, 'teacher_selection.csv'))
    if problems:
        print("[collate] PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("[collate] all expected runs present, recipe consistent")

    os.makedirs(args.out, exist_ok=True)
    summary = summarize(rows, args.mwpm)
    write_csv(os.path.join(args.out, 'gru_compression_p004_summary.csv'), summary,
              list(summary[0].keys()))

    long_rows = [{'role': r['_role'], 'units': r['_units'], 'params': r['_params'],
                  'mode': r['_mode'], 'seed': r['_seed'], 'p_L': r['p_L'],
                  'best_val_loss': r.get('best_val_loss', ''),
                  'epochs_ran': r.get('epochs_ran', ''),
                  'agree_ambiguous': r.get('agree_ambiguous', ''),
                  'tag': r['_tag']} for r in rows]
    long_rows.sort(key=lambda r: (-r['params'], r['mode'], r['seed']))
    write_csv(os.path.join(args.out, 'gru_compression_p004_by_seed.csv'), long_rows,
              list(long_rows[0].keys()))

    deltas = paired_differences(rows)
    if deltas:
        write_csv(os.path.join(args.out, 'gru_compression_p004_paired_delta.csv'), deltas,
                  list(deltas[0].keys()))

    print()
    print(f"  {'params':>8} {'units':>6} {'mode':>8} {'n':>2} {'mean p_L':>10} "
          f"{'std':>9} {'xMWPM':>7} {'train s':>9} {'epochs':>7} {'cap hit':>8}")
    for r in summary:
        std = f"{r['std_p_L']:.6f}" if r['std_p_L'] != '' else '-'
        print(f"  {r['params']:>8,} {r['units']:>6} {r['mode']:>8} {r['n_seeds']:>2} "
              f"{r['mean_p_L']:>10.6f} {std:>9} {r['ratio_vs_mwpm']:>7.3f} "
              f"{r['mean_train_time_s']:>9} {r['mean_epochs_ran']:>7} "
              f"{r['n_hit_cap']}/{r['n_seeds']:<6}")
    total_s = sum(float(r['mean_train_time_s']) * r['n_seeds']
                  for r in summary if r['mean_train_time_s'] != '')
    print(f"\n  total training time = {total_s / 3600:.2f} GPU-hours over "
          f"{sum(r['n_seeds'] for r in summary)} runs")
    print(f"  MWPM on [15.2M, 17.0M) = {args.mwpm}")

    if deltas:
        print("\n  distilled minus hard, paired by seed (negative = distillation better)")
        for d in deltas:
            std = f"{d['std_delta']:.6f}" if d['std_delta'] != '' else '-'
            print(f"  {d['params']:>8,} params  n={d['n_paired_seeds']}  "
                  f"mean {d['mean_delta']:+.6f}  std {std}   [{d['deltas']}]")


if __name__ == '__main__':
    main()
