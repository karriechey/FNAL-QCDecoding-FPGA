# Crumble live-demo rehearsal guide

> **Doc date: 2026-06-19. Still valid, but not a result** — it's a presentation/demo guide,
> not an experiment record. No paper numbers here.

Purpose (keep honest): Crumble shows the **structure** the reference architecture's architecture is built to
exploit — *not* his network running. We inject single errors and watch detectors light up to
show (a) **spatial locality** = what his convolutional kernels capture, and (b) **temporal
spread across rounds** = what his recurrent layers capture. Lands as: *a flat decoder throws
this structure away; his architecture is built for it → that's why it's worth quantizing.*

Demo circuit: `surface_code:rotated_memory_z`, **d=5, r=5**, the 3 dataset noise channels
(viz uses r=5, not the trained r=2, so temporal spread is visible). File: `plots/crumble.html`.

---

## The "stuck in timeline view" blocker — resolved

Checked the generated circuit and Crumble's own JS:
- The circuit carries full 2D `QUBIT_COORDS` (5×5 data grid) **and** `POLYGON` stabilizer
  plaquettes, so the spatial layout is fully encoded.
- Crumble's **main canvas IS the 2D spatial view** by default. "Timeline" in the UI is a
  secondary focusable widget (`btnTimelineFocus`), not the default mode — there is no mode you
  get "stuck" in. Opening `plots/crumble.html` in a browser shows the surface-code grid.

So this was likely the Jupyter inline iframe being small/awkward, not Crumble itself. **Action:
open the standalone `plots/crumble.html` in a real browser (not the notebook).** Still verify on
the actual projector — Crumble's canvas sizing/zoom can differ there.

## Controls (verified from Crumble's JS)
- **`E`** = step forward one layer (time slice). **`Q`** = step back one layer.
- **Shift+`E`** / **Shift+`Q`** = jump 5 layers. **Home** / **End** = first / last layer.
- Click a qubit to select; drag to box-select.
- Placing a propagating Pauli error live: **confirm the exact key in Crumble during rehearsal**
  (Crumble's Pauli/marker hotkeys are not asserted here). If live injection is fiddly on the
  projector, fall back to the backup figure below — it shows the identical signatures.

---

## The two beats (signatures verified in Stim, noiseless d=5 so only the injected error fires)

**Beat 1 — spatial locality (data-qubit error).**
Inject one error on the central **data** qubit at coord **(5,5)** (Crumble qubit index 27).
→ exactly **2 detectors fire, both in the same round**, at the two diagonally-adjacent
stabilizers (4,4) and (6,6). Step with `E`: the detections appear together at one round.
*Say: "one physical error → a tight, local cluster of detections. That spatial pattern is what
a convolutional kernel is built to read."*

**Beat 2 — temporal spread (measurement/ancilla error).**
Inject one error on the central **ancilla** qubit at coord **(4,4)** (Crumble qubit index 26).
→ the **same** stabilizer at (4,4) fires in **two consecutive rounds** (a time-like pair). Step
with `E` and watch the same location light up round-over-round.
*Say: "a measurement error isn't localized in time — the same detector fires across rounds. A
flat decoder sees two unrelated bits; a recurrent layer sees one correlated event in time."*

Close: *"Spatial clusters + temporal correlations are exactly the two structures the reference architecture's RCNN
is built to exploit — and why quantizing **that** architecture, not a flat stand-in, is the goal."*

---

## Backup screenshots
`plots/crumble_demo_backup.png` — 2 rows × 3 rounds, rendered directly from Stim detector
samples (so it matches what Crumble shows): top row = data error (2 adjacent detectors, one
round); bottom row = measurement error (same stabilizer, two consecutive rounds). Drop this in
if the live demo fails on the projector.

Regenerate / re-derive signatures: the script lives in this session's history; key facts —
data error @(5,5)→dets @(4,4),(6,6) same round; meas error @(4,4)→det @(4,4) two consecutive
rounds. Noiseless circuit (`p=0`) is what makes the signature clean.
