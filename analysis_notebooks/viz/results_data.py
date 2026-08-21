# Created: 2026-08-21
# Every scored configuration in Experiment 16, in one place. The figures, the interactive
# page and the animation all read from here, so a number corrected here is corrected
# everywhere.
#
# ratio is p_L divided by the MWPM decoded on that run's own evaluation shots.
# s_epoch is the median epoch time from the run log, and reflects card contention.

MWPM = {5: 0.007571, 7: 0.004291, 9: 0.002267}      # grace1 pools, [15.2M, 17.0M)
MWPM_SEALED = 0.007501                              # d=5, [17M, 19M), 2M shots

RUNS = [
    # d, arch, label,       params, batch, shots, seeds, p_L,      sd,       mwpm,     s_epoch, minutes, machine, date
    (5, 'gru', 'GRU u140',   69441, 10000, 10e6, 3, 0.008565, 0.000191, 0.007571,  8,  27, 'GH200', '08-19'),
    (5, 'gru', 'GRU u140',   69441, 10000, 15e6, 3, 0.007512, 0.000025, 0.007571, 17,  58, 'GH200', '08-21'),
    (5, 'gru', 'GRU u140',   69441, 10000, 20e6, 3, 0.007167, 0.000070, 0.007571, 16,  55, 'GH200', '08-21'),
    (5, 'gru', 'GRU u200',  135201, 10000, 10e6, 1, 0.008579, 0.0,      0.007571, 16,  53, 'GH200', '08-21'),
    (5, 'gru', 'GRU u280',  256481, 10000, 10e6, 1, 0.008975, 0.0,      0.007571, 16,  53, 'GH200', '08-21'),
    (5, 'mlp', 'MLP 209',    69389, 10000, 10e6, 3, 0.010509, 0.000071, 0.007571,  2,   7, 'GH200', '08-19'),
    (5, 'rcnn', 'RCNN',      69485,  5000, 10e6, 1, 0.008905, 0.0,      0.007572, 697, 396, 'A100', '08-14'),
    (5, 'rcnn', 'RCNN',      69485,  5000, 15e6, 1, 0.011098, 0.000455, 0.007572, 1050, 700, 'A100', '08-14'),
    (7, 'gru', 'GRU u140',   79521, 10000,  2e6, 3, 0.041780, 0.002883, 0.004291,  5,  17, 'GH200', '08-20'),
    (7, 'gru', 'GRU u140',   79521, 10000,  5e6, 3, 0.022014, 0.001207, 0.004291,  5,  17, 'GH200', '08-20'),
    (7, 'gru', 'GRU u140',   79521, 10000, 10e6, 3, 0.012324, 0.000447, 0.004291, 11,  37, 'GH200', '08-19'),
    (7, 'gru', 'GRU u140',   79521, 10000, 15e6, 3, 0.009883, 0.001088, 0.004291, 15,  50, 'GH200', '08-21'),
    (7, 'gru', 'GRU u140',   79521, 10000, 20e6, 2, 0.009045, 0.000921, 0.004291, 15,  50, 'GH200', '08-21'),
    (7, 'gru', 'GRU u280',  276641, 10000, 10e6, 1, 0.013338, 0.0,      0.004291,  8,  27, 'GH200', '08-20'),
    (7, 'mlp', 'MLP 160',    79841, 10000, 10e6, 3, 0.041561, 0.004435, 0.004291,  5,  17, 'GH200', '08-19'),
    (9, 'gru', 'GRU u140',   92961, 10000, 10e6, 3, 0.028793, 0.001338, 0.002267, 10,  35, 'GH200', '08-19'),
    (9, 'gru', 'GRU u280',  303521, 10000, 10e6, 3, 0.024116, 0.000919, 0.002267, 11,  37, 'GH200', '08-21'),
    # Diverged: val_loss bottomed at 0.0796 (epoch 56) then rose to 0.249. The scored
    # checkpoint is that epoch-56 model, worse than every 10M seed, so this is a record of a
    # failed optimisation rather than a data-scaling point. DIVERGED marks it for the plots.
    (9, 'gru', 'GRU u140',   92961, 10000, 15e6, 1, 0.030758, 0.0,      0.002267, 16,  53, 'GH200', '08-21'),
    # LR 1e-3 instead of the protocol's 3e-3: the lower rate removed the instability, but it
    # changes two variables against the 10M baseline, so it is a diagnostic, not a ladder rung.
    (9, 'gru', 'GRU u140 LR1e-3', 92961, 10000, 10e6, 1, 0.029258, 0.0, 0.002267,  8, 28, 'GH200', '08-21'),
    (9, 'gru', 'GRU u140 LR1e-3', 92961, 10000, 15e6, 1, 0.027671, 0.0, 0.002267, 12, 53, 'GH200', '08-21'),
    (9, 'rcnn', 'RCNN',      76117,  5000, 10e6, 1, 0.074773, 0.0,      0.002303, 8650, 6060, 'A100', '08-14'),
]
COLS = ('d', 'arch', 'label', 'params', 'batch', 'shots', 'seeds', 'p_L', 'sd',
        'mwpm', 's_epoch', 'minutes', 'machine', 'date')

# Runs whose optimisation failed. They are real measurements of what the recipe did, and
# they belong in the tables, but a ladder line drawn through them would imply the data
# volume caused the result when the optimiser did.
DIVERGED = {(9, 'GRU u140', 15e6)}

# The sealed-block confirmation of the winning configuration, kept separate: different
# shots, different MWPM, and it is a confirmation rather than a tuning result.
SEALED = dict(d=5, label='GRU u140 20M aug', params=69441, shots=20e6, seeds=3,
              p_L=0.007141, sd=0.000083, mwpm=MWPM_SEALED, block='[17M, 19M)')

def rows(include_diverged=True):
    out = [dict(zip(COLS, r)) for r in RUNS]
    for r in out:
        r['diverged'] = (r['d'], r['label'], r['shots']) in DIVERGED
    return out if include_diverged else [r for r in out if not r['diverged']]


def load_curves(meta, epochs=200):
    """Every run whose history is complete and whose checkpoint was actually scored.

    Admission needs two things, both checkable from the synced metadata: a history with the
    full epoch budget, and a row in that seed's eval CSV naming this run's checkpoint. The
    COMPLETE marker is stricter still, but it only exists for runs launched after it was
    introduced, so keying on it here would silently drop the earlier ladder rungs.
    """
    import csv, glob, json, os
    out = []
    for h in sorted(glob.glob(os.path.join(meta, '*', 'seed*', 'runs', '*.history.json'))):
        seed_dir = os.path.dirname(os.path.dirname(h))
        tag = os.path.basename(h).replace('.history.json', '')
        hist = json.load(open(h))
        val = hist.get('val_loss', [])
        if len(val) != epochs:
            continue
        ev = os.path.join(seed_dir, 'eval_best_ckpt.csv')
        if not os.path.exists(ev):
            continue
        want = tag + '.best.weights.h5'
        if not any(r['weights'] == want for r in csv.DictReader(open(ev))):
            continue
        d = 5 if '_d5_' in tag else 7 if '_d7_' in tag else 9
        units = next((u for u in (200, 280, 140) if f'_u{u}_' in tag), 140)
        hidden = tag.split('_h')[1].split('_')[0] if tag.startswith('mlp') else ''
        shots = 20_000_000 if 'ext5000000' in tag else int(tag.split('_ntr')[1].split('_')[0])
        run = os.path.basename(os.path.dirname(seed_dir))   # .../<run dir>/seedN
        out.append(dict(
            run=run, lr=1e-3 if 'lr1e3' in run else 3e-3,
            tag=tag, d=d, arch='mlp' if tag.startswith('mlp') else 'gru',
            units=units, hidden=hidden, shots=shots,
            seed=int(tag.split('_seed')[1].split('_')[0]),
            val=[float(x) for x in val], train=[float(x) for x in hist.get('loss', [])],
            best=min(range(len(val)), key=lambda i: val[i]),
            verified=os.path.exists(os.path.join(seed_dir, 'COMPLETE'))))
    # One line per (run directory, tag). The tag alone is not unique: it carries no learning
    # rate, so a run repeated at a different rate collides with its original, and keying on it
    # would silently drop one of the two.
    seen, uniq = set(), []
    for c in sorted(out, key=lambda c: (c['d'], c['shots'], c['units'], c['seed'])):
        key = (c['run'], c['tag'])
        if key in seen:
            continue
        seen.add(key); uniq.append(c)
    return uniq


# The RCNN runs that belong to the p=0.004 ladder of record, named explicitly. The EAF
# directory also holds rungs trained on the superseded 14M pool (base rate 0.19286) and a
# collapsed d=9 run that scored exactly its base rate, and neither is a point on this
# ladder -- an allowlist is safer here than a glob.
RCNN_CURVES = [
    ('out_d5_p004_ladder/ntr10000000_seed0/rcnn_d5_p0.004_r5_seed0_ntr10000000.history.json',
     5, 10_000_000, 0),
    ('out_d5_p004_ladder/ntr15000000_seed0/rcnn_d5_p0.004_r5_seed0_ntr15000000.history.json',
     5, 15_000_000, 0),
    ('out_d5_p004_ladder/ntr15000000_seed1/rcnn_d5_p0.004_r5_seed1_ntr15000000.history.json',
     5, 15_000_000, 1),
    ('out_d9_p004_ladder/ntr10000000_seed0/rcnn_d9_p0.004_r9_seed0_ntr10000000.history.json',
     9, 10_000_000, 0),
]


def load_rcnn_curves(meta_eaf):
    """RCNN validation curves from the EAF ladder. Empty when the directory is absent.

    These stop where early stopping fired -- 34 to 42 epochs against the students' fixed
    200 -- which is a real difference in protocol, not a truncated file.
    """
    import json, os
    out = []
    for rel, d, shots, seed in RCNN_CURVES:
        path = os.path.join(meta_eaf, rel)
        if not os.path.exists(path):
            continue
        val = json.load(open(path)).get('val_loss', [])
        if not val:
            continue
        out.append(dict(tag=os.path.basename(path).replace('.history.json', ''),
                        d=d, arch='rcnn', units=0, hidden='', shots=shots, seed=seed,
                        val=[float(x) for x in val], train=[],
                        best=min(range(len(val)), key=lambda i: val[i]), verified=True))
    return out
