#!/usr/bin/env python3
# Created: 2026-09-22
# Last modified: 2026-09-22
"""Preflight and teacher gate for the 20M GRU distillation study (d=5, r=5, p=0.004).

Run twice by eaf_run_gru_distill_20m.sh, before any student trains.

  --stage inputs
      Resolves and prints every input with its hash: primary pool, extension pool (both
      fingerprints and recomputed flips SHA-256), teacher checkpoint SHA-256, the four
      primary-pool blocks and their disjointness, and the measured parameter count of the
      teacher and every student width. Optionally compares against expected hashes, so a
      regenerated pool or a corrupted upload is caught before any compute.

  --stage caches
      After dump_teacher_probs.py has written the four teacher caches, verifies each one
      against its pool with the same checks the training script applies (flips match,
      contiguous coverage, same checkpoint behind all four, genuine non-constant output),
      then runs the teacher gate:

          teacher p_L on the evaluation block
          MWPM p_L decoded on the same shots
          ratio, and the paired McNemar test

      Exit code 0 when the teacher's p_L is below MWPM's on this exact block, 3 when it is
      not. The launcher stops on anything other than 0, so no student trains against a
      teacher that does not beat MWPM on the shots this study reports.

Writes (both stages) a JSON record into --out-json, which the launcher keeps in the
experiment folder.
"""
import argparse
import json
import os
import sys

import numpy as np

from train_gru_distillation_20m import (check_partitions, load_teacher_block, read_pool,
                                        sha256_of)

LOG = '[preflight]'


def stage_inputs(a, record):
    zm, meta_m, sha_m = read_pool(a.main_pool, 'primary', a.d, a.rounds, a.p)
    n_main_pool = int(zm['flips'].shape[0])
    check_partitions(n_main_pool, a.main_n_train, a.val_start, a.val_n, a.eval_start,
                     a.eval_n, a.sealed_start)
    zx, meta_x, sha_x = read_pool(a.extra_pool, 'extension', a.d, a.rounds, a.p)
    n_extra_pool = int(zx['flips'].shape[0])
    problems = []
    if meta_x['gen_seed'] == meta_m['gen_seed']:
        problems.append(f"extension gen_seed {meta_x['gen_seed']} equals the primary's")
    if a.extra_n > n_extra_pool:
        problems.append(f"extension holds {n_extra_pool:,} shots, {a.extra_n:,} requested")
    if a.expect_main_flips_sha and sha_m != a.expect_main_flips_sha:
        problems.append(f"primary flips SHA {sha_m[:16]} != expected "
                        f"{a.expect_main_flips_sha[:16]} (not the pool of record)")
    if a.expect_extra_flips_sha and sha_x != a.expect_extra_flips_sha:
        problems.append(f"extension flips SHA {sha_x[:16]} != expected "
                        f"{a.expect_extra_flips_sha[:16]}")

    w_sha = sha256_of(a.teacher_weights)
    if not w_sha:
        problems.append(f"teacher weights missing: {a.teacher_weights}")
    elif a.teacher_sha256 and w_sha != a.teacher_sha256:
        problems.append(f"teacher weights SHA {w_sha[:16]} != expected "
                        f"{a.teacher_sha256[:16]} (upload corrupted or wrong file)")
    print(f"{LOG} teacher checkpoint {a.teacher_weights}", flush=True)
    print(f"{LOG}   sha256={w_sha or 'MISSING'}", flush=True)

    # Measured parameter counts. build_gru_student imports TensorFlow; CPU is enough.
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    from StudentModels import build_gru_student, detector_sequence_layout
    n_t, n_pos, _ = detector_sequence_layout(a.d, a.rounds, a.p)
    counts = {}
    for u in [a.teacher_units] + a.student_units:
        m = build_gru_student(d=a.d, rounds=a.rounds, inputs='evts', units=u, hidden=(),
                              p=a.p)
        counts[u] = int(m.count_params())
        role = 'teacher' if u == a.teacher_units else 'student'
        print(f"{LOG}   {role:7s} GRU units={u:4d}  params={counts[u]:,}  "
              f"reset_after={m.get_layer('gru').reset_after}", flush=True)
    if w_sha:
        # Prove the checkpoint loads into the architecture it is claimed to be.
        t = build_gru_student(d=a.d, rounds=a.rounds, inputs='evts', units=a.teacher_units,
                              hidden=(), p=a.p)
        t.load_weights(a.teacher_weights)
        print(f"{LOG}   teacher checkpoint loads into GRU units={a.teacher_units} "
              f"(input {n_t} timesteps x {n_pos} positions)", flush=True)

    record.update(
        main_pool=os.path.abspath(a.main_pool), main_pool_flips_sha256=sha_m,
        main_pool_gen_seed=meta_m['gen_seed'], main_pool_n=n_main_pool,
        extra_pool=os.path.abspath(a.extra_pool), extra_pool_flips_sha256=sha_x,
        extra_pool_gen_seed=meta_x['gen_seed'], extra_pool_n=n_extra_pool,
        extra_pool_purpose=meta_x.get('purpose', ''),
        teacher_weights=os.path.abspath(a.teacher_weights), teacher_weights_sha256=w_sha,
        train_main=[0, a.main_n_train], train_extra=[0, a.extra_n],
        validation=[a.val_start, a.val_start + a.val_n],
        evaluation=[a.eval_start, a.eval_start + a.eval_n],
        sealed=[a.sealed_start, n_main_pool],
        param_counts={str(k): v for k, v in counts.items()})
    return problems


def stage_caches(a, record):
    zm, _meta_m, sha_m = read_pool(a.main_pool, 'primary', a.d, a.rounds, a.p)
    zx, _meta_x, sha_x = read_pool(a.extra_pool, 'extension', a.d, a.rounds, a.p)
    fm = zm['flips'].astype(np.int8).reshape(-1)
    fx = zx['flips'].astype(np.int8).reshape(-1)
    v0, v1 = a.val_start, a.val_start + a.val_n
    e0, e1 = a.eval_start, a.eval_start + a.eval_n

    print(f"{LOG} teacher caches:", flush=True)
    blocks = [
        ('main-train', a.cache_main, sha_m, fm[0:a.main_n_train], 0, a.main_n_train),
        ('extra-train', a.cache_extra, sha_x, fx[0:a.extra_n], 0, a.extra_n),
        ('validation', a.cache_val, sha_m, fm[v0:v1], v0, v1),
        ('evaluation', a.cache_eval, sha_m, fm[e0:e1], e0, e1),
    ]
    loaded = {}
    for label, path, sha, flips, lo, hi in blocks:
        loaded[label] = load_teacher_block(path, label, sha, flips, lo, hi, a.teacher_units)
    shas = {v['stats']['teacher_weights_sha256'] for v in loaded.values()}
    problems = []
    if len(shas) != 1:
        problems.append(f"caches come from {len(shas)} different checkpoints")
    w_sha = sha256_of(a.teacher_weights)
    if w_sha and w_sha not in shas:
        problems.append(f"teacher weights on disk {w_sha[:16]} match no cache")
    if loaded['extra-train']['logit'].shape[0] != a.extra_n:
        problems.append("extension cache count differs from --extra-n")
    record['caches'] = {k: v['stats'] for k, v in loaded.items()}
    if problems:
        return problems

    # --- teacher gate on the exact evaluation block -----------------------------------
    import pymatching
    from eval_on_tail import build_circuit, mcnemar_from_correct
    truth = fm[e0:e1]
    det = zm['det_evts'][e0:e1].astype(np.int8)
    dem = build_circuit(a.d, a.p, a.rounds).detector_error_model(decompose_errors=True)
    m_pred = pymatching.Matching.from_detector_error_model(dem).decode_batch(
        det, bit_packed_predictions=False, bit_packed_shots=False).astype(np.int8).reshape(-1)
    m_correct = m_pred == truth
    t_pred = (loaded['evaluation']['logit'] > 0.0).astype(np.int8)
    t_correct = t_pred == truth
    t_pL, m_pL = float((~t_correct).mean()), float((~m_correct).mean())
    n = e1 - e0
    t_se = (t_pL * (1 - t_pL) / n) ** 0.5
    m_se = (m_pL * (1 - m_pL) / n) ** 0.5
    mc = mcnemar_from_correct(t_correct, m_correct)
    ratio = t_pL / m_pL
    beats = t_pL < m_pL
    print(f"{LOG} ================ TEACHER GATE ================", flush=True)
    print(f"{LOG} evaluation block [{e0:,}, {e1:,})  n={n:,}", flush=True)
    print(f"{LOG}   teacher p_L = {t_pL:.6f} +/- {t_se:.6f}", flush=True)
    print(f"{LOG}   MWPM    p_L = {m_pL:.6f} +/- {m_se:.6f}", flush=True)
    print(f"{LOG}   teacher / MWPM = {ratio:.4f}x", flush=True)
    print(f"{LOG}   McNemar: teacher-only right={mc['rcnn_only']:,}  MWPM-only right="
          f"{mc['mwpm_only']:,}  net={mc['net_rcnn_wins']:+,}  p_exact={mc['p_exact']:.3e}",
          flush=True)
    print(f"{LOG}   teacher beats MWPM on this block: {'YES' if beats else 'NO'}", flush=True)
    record['teacher_gate'] = dict(
        eval_range=[e0, e1], teacher_p_L=round(t_pL, 6), teacher_se=round(t_se, 6),
        mwpm_p_L=round(m_pL, 6), mwpm_se=round(m_se, 6), ratio=round(ratio, 4),
        teacher_only_right=mc['rcnn_only'], mwpm_only_right=mc['mwpm_only'],
        net_teacher_wins=mc['net_rcnn_wins'], mcnemar_p_exact=mc['p_exact'],
        teacher_beats_mwpm=bool(beats))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['inputs', 'caches'], required=True)
    ap.add_argument('--d', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--p', type=float, default=0.004)
    ap.add_argument('--main-pool', required=True)
    ap.add_argument('--extra-pool', required=True)
    ap.add_argument('--main-n-train', type=int, default=15_000_000)
    ap.add_argument('--extra-n', type=int, default=5_000_000)
    ap.add_argument('--val-start', type=int, default=15_000_000)
    ap.add_argument('--val-n', type=int, default=200_000)
    ap.add_argument('--eval-start', type=int, default=15_200_000)
    ap.add_argument('--eval-n', type=int, default=1_800_000)
    ap.add_argument('--sealed-start', type=int, default=17_000_000)
    ap.add_argument('--teacher-weights', required=True)
    ap.add_argument('--teacher-units', type=int, default=140)
    ap.add_argument('--teacher-sha256', default='')
    ap.add_argument('--student-units', type=int, nargs='*', default=[100, 70, 46, 34])
    ap.add_argument('--expect-main-flips-sha', default='')
    ap.add_argument('--expect-extra-flips-sha', default='')
    ap.add_argument('--cache-main', default=None)
    ap.add_argument('--cache-extra', default=None)
    ap.add_argument('--cache-val', default=None)
    ap.add_argument('--cache-eval', default=None)
    ap.add_argument('--out-json', required=True)
    a = ap.parse_args()

    record = {'stage': a.stage}
    problems = stage_inputs(a, record) if a.stage == 'inputs' else stage_caches(a, record)
    record['problems'] = problems
    with open(a.out_json, 'w') as fh:
        json.dump(record, fh, indent=2, default=str)
    print(f"{LOG} wrote {a.out_json}", flush=True)
    if problems:
        for pr in problems:
            print(f"{LOG} PROBLEM: {pr}", flush=True)
        print(f"{LOG} STOP.", flush=True)
        sys.exit(2)
    if a.stage == 'caches' and not record['teacher_gate']['teacher_beats_mwpm']:
        print(f"{LOG} teacher does NOT beat MWPM on this evaluation block. STOP before "
              "training students.", flush=True)
        sys.exit(3)
    print(f"{LOG} stage {a.stage}: all checks passed", flush=True)


if __name__ == '__main__':
    main()
