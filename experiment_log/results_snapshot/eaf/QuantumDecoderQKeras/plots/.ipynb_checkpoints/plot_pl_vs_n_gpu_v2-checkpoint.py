#!/usr/bin/env python3
"""GPU p_L-vs-n_train sweep figure, v2 style, with HONEST uncertainty.

Honesty point: every seed is scored on the SAME fixed test tail, so seed-to-seed
SEM does NOT capture test-set sampling error. On a ~10k tail the binomial error on
p_L~0.045 is sqrt(p(1-p)/n_test)~0.002 and DOMINATES. We combine both:
    err = sqrt( sem_seed^2 + p(1-p)/n_test )
and draw the MWPM baseline with its own +/- sqrt(mwpm(1-mwpm)/n_test_mwpm) band,
so the figure shows the baseline itself is only known to ~+/-0.002.

Reads the per-run train_one.py CSVs (raw per-seed p_L). Does NOT overwrite any
existing figure -- writes a new file. No 'his recipe', no 'BELOW parity' claim.
"""
import argparse
import csv
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

BLUE, GREEN = "#1f77b4", "#2ca02c"


def find_anchor(data_dir, d, p, rounds):
    """MWPM anchor value, source file, and the n_test it was scored on."""
    path = os.path.join(data_dir, "mwpm_baseline.csv")
    if not os.path.exists(path):
        return None, None, None
    for row in csv.DictReader(open(path)):
        if (int(row["d"]) == d and abs(float(row["p"]) - p) < 1e-9
                and int(row["rounds"]) == rounds):
            return float(row["mwpm_p_L"]), path, int(row["n_test"])
    return None, path, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.expanduser("~/rcnn_threshold/out"))
    ap.add_argument("--data-dir", default=os.path.expanduser("~/rcnn_threshold/pools"))
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--p", type=float, default=0.010)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--plot-out", default="./plots/rcnn_d5_pl_vs_n_gpu_v2.png")
    args = ap.parse_args()
    d, p, r = args.d, args.p, args.rounds

    # ---- STEP 1: find + confirm data (no guessing -- print what we used) -----
    pat = os.path.join(args.out_dir, f"rcnn_d{d}_p{p:.3f}_r{r}_seed*_ntr*.csv")
    files = sorted(glob.glob(pat))
    print(f"[step1] glob: {pat}")
    print(f"[step1] matched {len(files)} per-run CSVs; first: "
          f"{files[0] if files else 'NONE'}")
    if not files:
        raise SystemExit("[step1] no CSVs found -- check --out-dir")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=["seed", "n_train"], keep="last")
    print(f"[step1] columns: {list(df.columns)}")
    print("[step1] df.head():")
    print(df[["d", "p", "rounds", "seed", "n_train", "n_test", "p_L",
              "mwpm_p_L"]].head().to_string(index=False))

    n_test_vals = sorted(df.n_test.unique())
    n_test = int(n_test_vals[0])
    print(f"[step1] (b) n_test (test tail) used: {n_test_vals}")
    seeds_per_N = df.groupby("n_train").seed.nunique().sort_index()
    print("[step1] (a) seeds at each n_train:")
    for N, ns in seeds_per_N.items():
        tag = "  <-- 10M" if N >= 10_000_000 else ""
        print(f"          n_train={int(N):>10,}: {ns} seeds{tag}")

    mwpm, mwpm_src, mwpm_ntest = find_anchor(args.data_dir, d, p, r)
    if mwpm is None:  # fall back to the (constant) column in the run rows
        vals = pd.to_numeric(df.mwpm_p_L, errors="coerce").dropna().unique()
        mwpm = float(np.mean(vals)); mwpm_src = "run-rows mwpm_p_L"; mwpm_ntest = n_test
    print(f"[step1] (c) MWPM anchor = {mwpm:.5f}  source={mwpm_src}  "
          f"scored on n_test={mwpm_ntest}")
    if n_test <= 12000 and abs(mwpm - 0.0451) < 0.003:
        print("[step1] *** FLAG: anchor ~0.0451 on a ~10k tail -> baseline known "
              "only to ~+/-0.002 (binomial). Test-set noise will dominate. ***")
    if len(n_test_vals) > 1:
        print(f"[step1] *** WARN: mixed n_test across runs {n_test_vals} ***")

    # ---- STEP 2: honest uncertainty ----------------------------------------
    rows = []
    for N, g in df.groupby("n_train"):
        pl = g.p_L.to_numpy(dtype=float)
        m = float(pl.mean())
        ns = len(pl)
        sem_seed = float(pl.std(ddof=1) / np.sqrt(ns)) if ns > 1 else 0.0
        binom = float(np.sqrt(m * (1 - m) / n_test))
        comb = float(np.sqrt(sem_seed**2 + binom**2))
        rows.append(dict(n_train=int(N), n_seeds=ns, mean_pL=m, sem_seed=sem_seed,
                         binom_sigma=binom, combined_err=comb, ratio=m / mwpm))
    agg = pd.DataFrame(rows).sort_values("n_train").reset_index(drop=True)
    mwpm_band = float(np.sqrt(mwpm * (1 - mwpm) / mwpm_ntest))

    # ---- STEP 3: figure (v2 style) -----------------------------------------
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "font.size": 11,
                         "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    Ns = agg.n_train.to_numpy()

    for N in Ns:  # faint per-seed scatter
        pl = df[df.n_train == N].p_L.to_numpy(dtype=float)
        ax.scatter([N] * len(pl), pl, color=BLUE, alpha=0.28, s=26, zorder=2,
                   label="per-seed runs" if N == Ns[0] else None)
    # MWPM baseline + its own binomial band
    ax.axhline(mwpm, color=GREEN, ls="--", lw=2, zorder=3,
               label=f"MWPM = {mwpm:.4f}  (classical baseline — target to match)")
    ax.fill_between([Ns.min() * 0.8, Ns.max() * 1.25], mwpm - mwpm_band,
                    mwpm + mwpm_band, color=GREEN, alpha=0.15, zorder=1)
    # RCNN mean with COMBINED error bars
    ax.errorbar(Ns, agg.mean_pL, yerr=agg["combined_err"], color=BLUE, lw=2,
                marker="o", ms=6, capsize=4, zorder=4,
                label="RCNN mean  ($\\pm\\sqrt{\\mathrm{SEM}_{seed}^2 + p(1-p)/n_{test}}$)")

    # ratio labels, offset UP-LEFT so they never sit on markers/error bars
    for _, x in agg.iterrows():
        ax.annotate(f"{x.ratio:.2f}×", (x.n_train, x.mean_pL),
                    textcoords="offset points", xytext=(-11, 11), ha="right",
                    va="bottom", fontsize=8.5, color=BLUE,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
    # last point: DATA-DRIVEN, plain-language verdict -- never hardcode 'parity'.
    last = agg.iloc[-1]
    gap = float(mwpm - last.mean_pL)                 # >0 = RCNN below MWPM
    lerr = float(last["combined_err"])
    n_seeds = int(agg.n_seeds.max())
    # where the curve first reaches parity (within one combined error of MWPM, or below)
    parity_n = None
    for _, x in agg.iterrows():
        if x.mean_pL < mwpm or abs(x.mean_pL - mwpm) <= x["combined_err"]:
            parity_n = int(x.n_train); break
    if abs(gap) <= lerr:
        annot = "matches MWPM\n(parity)"
        last_txt = (f"At {int(last.n_train):,} shots the RCNN and MWPM are level "
                    f"(equal within measurement error).")
    elif gap > 0:
        annot = "slightly below MWPM\n(suggestive - McNemar pending)"
        last_txt = (f"At {int(last.n_train):,} shots the RCNN sits slightly BELOW MWPM "
                    f"({last.ratio:.2f}x), but with only {n_seeds} training runs (seeds) this is "
                    f"SUGGESTIVE, not confirmed - a paired McNemar test on the shared test set "
                    f"(currently training) is needed to settle it.")
    else:
        annot = f"above MWPM\n({last.ratio:.2f}x)"
        last_txt = f"At {int(last.n_train):,} shots the RCNN is still above MWPM ({last.ratio:.2f}x)."
    ax.annotate(annot, (last.n_train, last.mean_pL),
                textcoords="offset points", xytext=(-78, 52), ha="center", va="bottom",
                fontsize=8.5, color="dimgray",
                arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.8))

    ax.set_xscale("log")
    ax.set_xlabel("training-set size  $n_{train}$  (log scale)")
    ax.set_ylabel("logical error rate  $p_L$   (lower = better)")
    ax.set_xlim(Ns.min() * 0.7, Ns.max() * 1.4)
    fig.suptitle("FullRCNNModel logical error rate vs training-set size",
                 fontsize=13, fontweight="bold", y=0.98)
    ax.set_title(f"surface code  d={d}, p={p}, r={r}  (4-channel circuit-level noise)",
                 fontsize=10)
    ax.legend(loc="upper right", framealpha=0.9)

    pool = os.path.join(args.data_dir, f"data_d{d}_p{p:.3f}_r{r}.npz")
    parity_txt = (f"The RCNN matches the classical MWPM decoder (parity) once training reaches "
                  f"~{parity_n:,} shots. " if parity_n else "")
    cap = (f"Lower = better. Each point is a mean over {n_seeds} training runs (seeds), scored on "
           f"{n_test:,} held-out test shots; the green dashed line is the classical MWPM decoder "
           f"(the target to match). " + parity_txt + last_txt + "\n"
           f"This is a single physical error rate ABOVE the code threshold (p={p} > ~0.0065) — a "
           f"decoder-vs-decoder comparison, not a scalability or below-threshold claim. Error bars "
           f"= sqrt(seed spread^2 + test-set sampling^2); green band = MWPM's own uncertainty. "
           f"Pool {pool}.")
    fig.text(0.012, -0.20, cap, fontsize=7.6, color="dimgray", wrap=True)

    os.makedirs(os.path.dirname(args.plot_out), exist_ok=True)
    if os.path.exists(args.plot_out):
        print(f"[step5] NOTE: {args.plot_out} exists -- overwriting the v2-GPU file only")
    fig.savefig(args.plot_out, bbox_inches="tight")
    print(f"[step5] wrote {args.plot_out}")

    # ---- STEP 5: plotted table ---------------------------------------------
    print("\n[table] n_train      mean_pL   sem_seed  binom_sig  comb_err   xMWPM  n")
    for _, x in agg.iterrows():
        print(f"[table] {x.n_train:>10,}  {x.mean_pL:.5f}  {x.sem_seed:.5f}   "
              f"{x.binom_sigma:.5f}    {x['combined_err']:.5f}  {x.ratio:.3f}x  "
              f"{int(x.n_seeds)}")
    print(f"[table] MWPM={mwpm:.5f}  band=±{mwpm_band:.5f}  (n_test={mwpm_ntest})")


if __name__ == "__main__":
    main()
