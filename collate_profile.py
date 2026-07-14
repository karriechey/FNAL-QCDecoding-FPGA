#!/usr/bin/env python3
"""Collate the per-seed profile_ranges JSONs into the three tables Phase 2/4 need:

  (1) n-vs-nonpositivity: per CNNStateCorrelator instance, its state-count n against
      frac_nonpositive. Tests the structural hypothesis -- the phase-weighted quadratic
      form s^T C s is provably >=0 for n=2 (=(s1+c s2)^2 + s2^2(1-c^2)) but the 3x3+
      phase matrix (off-diagonals in (-1,1)) need NOT be PSD, so it can go <=0 for n>=3.
      If every n=2 instance is at 0.00% and only n>=3 go negative, the finding is
      STRUCTURAL ("the phase matrix is unconstrained -> indefinite for n>=3"), which names
      exactly which components need a fix and points at the PSD-parameterization option.
  (2) implied_I stability across seeds (a Table 3 caveat if any site's I moves).
  (3) the quantizable (bounded) sites = the actual Phase-2 activation-quantizer config.

NOTE these are STATIC observations; whether the log-domain fix is signed-LSE or a
PSD-constrained c_phi (retrain) is a design choice for Phase 4, not decided here.

  python collate_profile.py [--dir ~/rcnn_threshold/out_q_mcnemar] [--bits 6]
"""
import argparse, glob, json, os


def load_seeds(dir_, bits):
    out = {}
    for f in sorted(glob.glob(os.path.join(dir_, f'profile_ranges_w{bits}_seed*.json'))):
        s = json.load(open(f))
        seed = s.get('_provenance', {}).get('seed', os.path.basename(f))
        out[str(seed)] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.expanduser('~/rcnn_threshold/out_q_mcnemar'))
    ap.add_argument('--bits', type=int, default=6)
    a = ap.parse_args()
    seeds = load_seeds(a.dir, a.bits)
    if not seeds:
        raise SystemExit(f'no profile_ranges_w{a.bits}_seed*.json in {a.dir}')
    sk = sorted(seeds)
    print(f'seeds: {sk}')
    for s in sk:
        pv = seeds[s]['_provenance']
        print(f'  seed{s}: p_L {pv["profiled_p_L"]} (expected {pv["expected_p_L"]})')

    # (1) measured n  vs  lambda_min(C)<0 (CAUSE)  vs  combination<=0 (symptom) -- the structural test
    print('\n=== (1) CNNStateCorrelator: measured n | lambda_min(C)<0 (cause) | combination<=0 (symptom) ===')
    print(f'{"instance":22s} {"n":>3s}   ' + '  '.join(f'lam<0_s{s}' for s in sk)
          + '   ' + '  '.join(f'nonpos_s{s}' for s in sk))
    d0 = seeds[sk[0]]
    lam0 = d0.get('_correlator_lambda', {})
    corr = sorted(lam0, key=lambda k: int(k.split('#')[1]))
    def lam(inst, s):  # measured frac lambda_min<0 for this instance/seed
        return seeds[s].get('_correlator_lambda', {}).get(inst, {}).get('frac_lambda_min_neg', 0.0)
    def nonpos(inst, s):
        # CRASH (not 0.0) if the key is missing -- else a zero table prints "CONFIRMED" vacuously.
        key = f'combination_preclip@CNNModel.py:1896#{inst}'
        d = seeds[s].get(key)
        if d is None:
            raise SystemExit(f'seed{s}: missing {key} -- line moved / hook renamed? refusing to vacuously confirm')
        return d.get('frac_nonpositive', 0.0)
    for inst in corr:
        n = lam0[inst]['n_states']
        ns = '  ?' if n is None else f'{n:>3d}'    # None => correlator never fired (guard)
        print(f'{inst:22s} {ns}   ' + '  '.join(f'{lam(inst,s):7.2%}' for s in sk)
              + '   ' + '  '.join(f'{nonpos(inst,s):8.2%}' for s in sk))
    n2 = [i for i in corr if lam0[i]['n_states'] == 2]
    n3 = [i for i in corr if lam0[i]['n_states'] is not None and lam0[i]['n_states'] >= 3]
    max_n2 = max((lam(i, s) for i in n2 for s in sk), default=0)
    any_n3 = any(lam(i, s) > 0 for i in n3 for s in sk)
    # lambda_min<0 is NECESSARY for combination<=0, so frac(lam<0) >= frac(nonpos) must hold
    bound_ok = all(lam(i, s) + 1e-12 >= nonpos(i, s) for i in corr for s in sk)
    print(f'  n=2: {len(n2)} instances, max frac(lambda_min<0) = {max_n2:.2%}  (must be 0 -- proof of correct C assembly)')
    print(f'  n>=3: {len(n3)} instances, any indefinite = {any_n3}')
    print(f'  necessity bound frac(lam<0) >= frac(nonpos) holds: {bound_ok}')
    ok = (max_n2 < 1e-9) and any_n3 and bound_ok
    print(f'  STRUCTURAL RESULT (C is PSD for n=2, unconstrained/indefinite for n>=3): '
          f'{"CONFIRMED" if ok else "NOT clean -- inspect"}')

    # (2) implied_I stability across seeds
    print('\n=== (2) implied_I_from_max across seeds (Table 3 caveat if it MOVES) ===')
    keys = sorted(k for k, v in d0.items() if isinstance(v, dict) and 'implied_I_from_max' in v)
    moved = []
    for k in keys:
        Is = [seeds[s].get(k, {}).get('implied_I_from_max') for s in sk]
        if len(set(Is)) > 1:
            moved.append((k, Is))
    if moved:
        for k, Is in moved:
            print(f'  MOVES {k}: {Is}')
    else:
        print(f'  stable across {len(sk)} seeds for all {len(keys)} sites (report as: I is seed-stable)')

    # (3) quantizable bounded sites = the Phase-2 quantizer config
    print('\n=== (3) quantizable sites (frac@B6>=0) = Phase-2 activation-quantizer config ===')
    for k in sorted(d0):
        v = d0[k]
        if not (isinstance(v, dict) and 'implied_frac_at_B6' in v):
            continue
        if v['implied_frac_at_B6'] < 0 or 'preclip' in k:
            continue
        rr = v.get('rel_rmse_at_B6'); rr = 'n/a' if rr is None else f'{rr:.1%}'
        print(f'  {k:34s} {v["signed_source"]:8s} {"signed" if v["signed"] else "unsgn"}  '
              f'I={v["implied_I_from_max"]} frac@B6={v["implied_frac_at_B6"]} relRMSE@B6={rr}')


if __name__ == '__main__':
    main()
