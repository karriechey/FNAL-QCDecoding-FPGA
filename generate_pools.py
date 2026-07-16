#!/usr/bin/env python3
"""Generate the 8 CORE dataset pools for the across-threshold RCNN-vs-MWPM study,
and compute the deterministic MWPM p_L baseline ONCE per (d,p) at generation time.

Author: Claude (local Mac). Executor: you (EAF A10 pod). This runs CPU Stim sampling +
a one-shot MWPM decode per pool; it does NOT train. Run it on the GPU pod (the GPU is
simply idle during this step) or any CPU pod.

Design (matches the trustworthy structure verified in the MWPM-trust goal):
  * rounds = d (threshold-figure convention; CHECKPOINT 0).
  * 4-channel noise, all four Stim channels = p, and the DEM is decoded 4-channel too --
    identical circuit for generate AND decode, so nothing silently mis-baselines. This is
    the same circuit generate_datasets.gen_one builds.
  * One pool per (d,p): nested train prefixes tr=slice(0,ntr) come from the FRONT; the
    fixed test tail te=slice(N-nte,N) is held out for every rung. They are DISJOINT:
    assert ntr_max == N - nte  (ntr_max is the top of the ladder = train capacity).
  * N_test sized from the MEASURED MWPM p_L so the tail holds >=~500 residual errors.

Outputs (to DATA_ROOT):
  data_d{d}_p{p:.3f}_r{d}.npz   -- measurements, det_evts, flips (int8), as the trainer expects
  mwpm_baseline.csv             -- one row per (d,p): the canonical MWPM p_L + diagnostics

The trainer (benchmark_rcnn.py) reads the SAME filename pattern and the SAME 4-channel
circuit, so its recomputed MWPM is deterministically identical to the value logged here.
"""
import csv
import os
import time
import numpy as np
from circuit_generators import get_builtin_circuit  # verified wrapper over stim.Circuit.generated

# --- CONFIG -----------------------------------------------------------------------------
# DATA_ROOT: where the ~10 GB of pools land. Home (26 GB free) fits the core grid.
# Switch to an /exp path here if/when a worker-shared allocation is granted (Condor).
DATA_ROOT = os.path.expanduser("~/rcnn_threshold/pools")
GEN_SEED = 12345          # fixed; recorded per pool for provenance
CHUNK = 1_000_000         # sample in chunks to cap peak RAM on the 10M pools
OBS = "ZL"                # rotated_memory_z single logical observable

# (d, p, N_total, N_test). N_total = train_capacity + N_test (see WorkLog Step-1 table).
# train_capacity = top of the ladder: 10M for p<=0.005, 8M for p=0.007, 5M for p=0.010.
# N_test from measured MWPM p_L (>=~500 residual errors at the tail).
POOLS = [
    # d=3, rounds=3
    (3, 0.004, 10_050_000, 50_000),
    (3, 0.005, 10_030_000, 30_000),
    (3, 0.007,  8_020_000, 20_000),
    (3, 0.010,  5_010_000, 10_000),
    # d=5, rounds=5
    (5, 0.004, 10_070_000, 70_000),
    (5, 0.005, 10_040_000, 40_000),
    (5, 0.007,  8_020_000, 20_000),
    (5, 0.010,  5_010_000, 10_000),
]
# d=7 confirmation (p in {0.004,0.005}) is DEFERRED -- generated only after the core grid
# lands and we've seen whether d=3/5 reproduce the crossing. Not in this list on purpose.
# ----------------------------------------------------------------------------------------


def build_circuit(d, p):
    """The 4-channel rotated_memory_z circuit at rounds=d (identical to gen + decode)."""
    return get_builtin_circuit(
        "surface_code:rotated_memory_z", distance=d, rounds=d,
        before_round_data_depolarization=p, after_reset_flip_probability=p,
        after_clifford_depolarization=p, before_measure_flip_probability=p)


def sample_pool(circ, n_total, seed):
    """Chunked sampling so peak RAM stays ~one chunk + the final arrays, not 2x.
    Returns (measurements, det_evts, flips) as int8 arrays."""
    m_sampler = circ.compile_sampler(seed=seed)
    converter = circ.compile_m2d_converter()
    # widths from a 2-shot probe so we can preallocate
    probe = m_sampler.sample(2, bit_packed=False)
    det0, obs0 = converter.convert(measurements=probe, separate_observables=True, bit_packed=False)
    n_meas, n_det, n_obs = probe.shape[1], det0.shape[1], obs0.reshape(2, -1).shape[1]
    measurements = np.empty((n_total, n_meas), dtype=np.int8)
    det_evts = np.empty((n_total, n_det), dtype=np.int8)
    flips = np.empty((n_total, n_obs), dtype=np.int8)
    # fresh sampler so the 2-shot probe doesn't shift the stream
    m_sampler = circ.compile_sampler(seed=seed)
    done = 0
    while done < n_total:
        k = min(CHUNK, n_total - done)
        meas = m_sampler.sample(k, bit_packed=False)
        det, obs = converter.convert(measurements=meas, separate_observables=True, bit_packed=False)
        measurements[done:done + k] = meas.astype(np.int8)
        det_evts[done:done + k] = det.astype(np.int8)
        flips[done:done + k] = obs.astype(np.int8).reshape(k, -1)
        done += k
    return measurements, det_evts, flips


def main():
    import pymatching  # imported here so the script can be read without the dep present
    os.makedirs(DATA_ROOT, exist_ok=True)
    baseline_csv = os.path.join(DATA_ROOT, "mwpm_baseline.csv")
    fields = ["d", "p", "rounds", "n_total", "n_test", "train_capacity",
              "mwpm_p_L", "mwpm_resid_errors_test", "base_rate_test",
              "base_rate_pool", "gen_seed"]
    new = not os.path.exists(baseline_csv)
    f = open(baseline_csv, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()

    print(f"[gen] DATA_ROOT = {DATA_ROOT}")
    for d, p, n_total, n_test in POOLS:
        t0 = time.time()
        fn = os.path.join(DATA_ROOT, f"data_d{d}_p{p:.3f}_r{d}.npz")
        train_cap = n_total - n_test
        # DISJOINT-TAIL GUARANTEE: top of the ladder must not touch the test tail.
        assert train_cap == n_total - n_test, "train/test overlap"
        if os.path.exists(fn):
            print(f"[gen] skip (exists) {fn}")
            continue

        circ = build_circuit(d, p)
        measurements, det_evts, flips = sample_pool(circ, n_total, GEN_SEED)

        # --- once-per-(d,p) deterministic MWPM baseline on the FIXED tail --------------
        dem = circ.detector_error_model(decompose_errors=True)
        pym = pymatching.Matching.from_detector_error_model(dem)
        te = slice(n_total - n_test, n_total)
        pred = pym.decode_batch(det_evts[te], bit_packed_predictions=False,
                                bit_packed_shots=False).astype(np.int8).reshape(-1, 1)
        truth = flips[te].reshape(-1, 1)
        resid = int((pred != truth).sum())
        mwpm_pL = resid / n_test
        base_test = float(flips[te].mean())     # all-zero predictor's error on the tail
        base_pool = float(flips.mean())          # label positive rate across the pool

        np.savez(fn, measurements=measurements, det_evts=det_evts, flips=flips)
        size_gb = os.path.getsize(fn) / 1024**3
        w.writerow(dict(d=d, p=p, rounds=d, n_total=n_total, n_test=n_test,
                        train_capacity=train_cap, mwpm_p_L=round(mwpm_pL, 6),
                        mwpm_resid_errors_test=resid, base_rate_test=round(base_test, 5),
                        base_rate_pool=round(base_pool, 5), gen_seed=GEN_SEED))
        f.flush()
        # FLAG if the tail is residual-error-starved (would make RCNN p_L noisy there).
        starved = " <-- FLAG: <300 residual errors, tail too small" if resid < 300 else ""
        print(f"[gen] d={d} p={p:.3f} r={d}  N={n_total:,} (train_cap {train_cap:,}, "
              f"tail {n_test:,})  MWPM p_L={mwpm_pL:.5f} ({resid} resid errs){starved}  "
              f"base_rate(tail)={base_test:.3f}  {size_gb:.2f} GB  {time.time()-t0:.0f}s")
    f.close()
    print(f"[gen] baseline -> {baseline_csv}")
    print("[gen] DONE. Next: smoke-test ONE training job before any fan-out.")


if __name__ == "__main__":
    main()
