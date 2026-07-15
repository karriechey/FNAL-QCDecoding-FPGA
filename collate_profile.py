#!/usr/bin/env python3
# Created: 2026-07-14
# Last modified: 2026-07-15
"""Collate the per-seed profile_ranges JSONs into the tables the paper section needs:

  (1) n-vs-nonpositivity: per CNNStateCorrelator instance, measured n vs lambda_min(C)<0
      (cause) and combination<=0 (symptom). s^T C s is provably >=0 for n=2
      (=(s1+c s2)^2 + s2^2(1-c^2)) but the 3x3 phase matrix (off-diag in (-1,1)) need NOT
      be PSD, so it can go <0 for n>=3 -- the structural cause, names which components.
  (2) implied_I stability across seeds -- BOTH from_max (tail stat, expected to move) AND
      from_p99_9 (the I policy relies on this being stable; if it moves the policy changes).
  (3) x-like range table (LaTeX), all 3 seeds -- the un-quantizable kernel/correlator outputs.
  (4) fixed_point_format_table (LaTeX) -- the FINAL per-tensor ap_fixed<B,I> config that
      configures ActQuant._int_bits and goes in quantization.tex. signed from TYPE; I with
      provenance A (analytic: z-like 4, p/f 0, cphi/alpha 1) or P (profiled: decoder ReLU/logit,
      I from p99.9, MAX across seeds -- NOT seed 0).

NOTE STATIC observations; whether the log-domain fix is signed-LSE or PSD-constrained c_phi
(retrain) is a Phase-4 choice, not decided here.

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
    # lambda_min<0 is NECESSARY for combination<=0, so frac(lam<0) >= frac(nonpos) must hold.
    # Tolerance 1e-3: at n=2 the combination is >=0 exactly, but fp roundoff in the sum yields a
    # handful of ~-1e-15 values counted as nonpos (~1e-6 rate) against an exact-0 lambda count.
    # That fp dust is not a necessity violation; 1e-3 ignores it, real n>=3 gap is ~0.8.
    TOL = 1e-3
    viol = [(i, s, nonpos(i, s), lam(i, s)) for i in corr for s in sk if nonpos(i, s) > lam(i, s) + TOL]
    bound_ok = not viol
    if viol:
        for i, s, np_, lm in viol:
            print(f'  BOUND VIOLATION {i} seed{s}: nonpos={np_:.3%} > lam<0={lm:.3%} -- necessity broken!')
    print(f'  n=2: {len(n2)} instances, max frac(lambda_min<0) = {max_n2:.2%}  (must be 0 -- proof of correct C assembly)')
    print(f'  n>=3: {len(n3)} instances, any indefinite = {any_n3}')
    print(f'  necessity bound frac(lam<0) >= frac(nonpos) holds: {bound_ok}')
    ok = (max_n2 < 1e-9) and any_n3 and bound_ok
    print(f'  STRUCTURAL RESULT (C is PSD for n=2, unconstrained/indefinite for n>=3): '
          f'{"CONFIRMED" if ok else "NOT clean -- inspect"}')

    # (2) implied_I stability across seeds -- BOTH from_max and from_p99_9
    keys = sorted(k for k, v in d0.items() if isinstance(v, dict) and 'implied_I_from_max' in v)
    for field, label in (('implied_I_from_max', 'from_max (tail stat -- expected to move)'),
                         ('implied_I_from_p99_9', 'from_p99_9 (the I policy relies on THIS being stable)')):
        print(f'\n=== (2) implied_I {label} across seeds ===')
        moved = []
        for k in keys:
            Is = [seeds[s].get(k, {}).get(field) for s in sk]
            if len(set(Is)) > 1:
                moved.append((k, Is))
        if moved:
            print(f'  {len(moved)}/{len(keys)} sites MOVE:')
            for k, Is in moved:
                print(f'    MOVES {k}: {Is}')
        else:
            print(f'  STABLE across {len(sk)} seeds for all {len(keys)} sites')
        # bounded (frac@B6>=0) sites are the ones that matter for the config -- report them separately
        bmoved = [(k, Is) for k, Is in moved if d0.get(k, {}).get('implied_frac_at_B6', -9) >= 0]
        print(f'  of the BOUNDED (frac@B6>=0) config sites: {len(bmoved)} move '
              f'(by at most +-{max((max(Is)-min(Is) for _, Is in bmoved), default=0)} bit)')

    def out_sites(pred):
        return sorted({k for s in sk for k, v in seeds[s].items()
                       if isinstance(v, dict) and 'implied_I_from_max' in v and pred(k)},
                      key=lambda k: (k.split('#')[0], int(k.split('#')[1].split('_')[0]) if '#' in k else 0))
    def field(k, s, f):
        return seeds[s].get(k, {}).get(f)

    # (3) x-like range table (LaTeX), all 3 seeds -- the un-quantizable kernel/correlator outputs
    xl = out_sites(lambda k: ('CNNKernelWithEmbedding#' in k or 'CNNStateCorrelator#' in k) and k.endswith('_out'))
    print('\n=== (3) x-like range table (LaTeX) -- un-quantizable (Phase 4) ===')
    print('% tensor & seed & min & max & I(max) & frac@B6')
    print('\\begin{tabular}{llrrrr}\\hline tensor & seed & min & max & $I$ & frac@$B{=}6$ \\\\\\hline')
    for k in xl:
        for s in sk:
            v = seeds[s].get(k, {})
            if not v:
                continue
            print(f'{k.replace("_out","").replace("#","\\#")} & {s} & {v["min"]:.2g} & {v["max"]:.2g} '
                  f'& {v["implied_I_from_max"]} & {v["implied_frac_at_B6"]} \\\\')
    print('\\hline\\end{tabular}')

    # (4) fixed_point_format_table (LaTeX) -- FINAL per-tensor config by TYPE (not seed 0).
    # A = analytic bound; P = profiled (I from p99.9, MAX across seeds). Where analytic disagrees
    # with the profile (analytic too small), we FLAG and use the conservative profiled I.
    def classify(k):   # -> (type, signed, analytic_I or None)
        if k == 'dec_in':                              return ('z-like', True, 4)
        if k.startswith('dec_layer') and 'relu' in k:  return ('relu', False, None)   # profiled
        if k.startswith('dec_layer') and 'sigmoid' in k: return ('p/f', False, 0)
        if 'TripletStateProbEmbedder' in k:            return ('p/f', False, 0)
        if 'DetectorBitStateEmbedder' in k or 'DetectorEventStateEmbedder' in k: return ('cphi/alpha', True, 1)
        if 'Combiner#' in k and k.endswith('_out'):    return ('z-like', True, 4)      # post-log z''
        return (None, None, None)                      # x-like / other -> not a config row
    cfg = out_sites(lambda k: classify(k)[0] is not None)
    print('\n=== (4) fixed_point_format_table (LaTeX) -- FINAL config (configures ActQuant._int_bits) ===')
    print('% tensor & type & signed & I & prov & frac@B6 & relRMSE@B6')
    print('\\begin{tabular}{llccclr}\\hline tensor & type & sgn & $I$ & prov & frac@$B{=}6$ & relRMSE@$B{=}6$ \\\\\\hline')
    flags = []
    for k in cfg:
        typ, signed, aI = classify(k)
        pI = max(field(k, s, 'implied_I_from_p99_9') or 0 for s in sk)   # profiled, conservative
        if aI is None:                       # relu/logit: always profiled
            I, prov = pI, 'P'
        elif aI >= pI:                       # analytic bound covers the data
            I, prov = aI, 'A'
        else:                                # analytic assumption VIOLATED by data -> use profiled
            I, prov = pI, 'P!'; flags.append((k, typ, aI, pI))
        frac = 6 - I - int(signed)
        rr = max((field(k, s, 'rel_rmse_at_B6_Ip999') or 0) for s in sk)
        print(f'{k.replace("_out","").replace("#","\\#")} & {typ} & {"S" if signed else "U"} & {I} & {prov} '
              f'& {frac} & {rr:.1%} \\\\')
    print('\\hline\\end{tabular}')
    if flags:
        print('\n  ANALYTIC-BOUND VIOLATIONS (writeup taxonomy too small; profiled I used):')
        for k, typ, aI, pI in flags:
            print(f'    {k}: type {typ} expects I={aI}, profiled I={pI} across seeds -> use {pI}')


if __name__ == '__main__':
    main()
