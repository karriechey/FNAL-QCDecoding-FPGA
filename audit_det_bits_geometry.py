#!/usr/bin/env python3
# Created: 2026-08-12
# Last updated: 2026-08-12
"""Consistency check for the det_bits -> kernel grouping used by FullRCNNModel.

At d=5, r=5 the RCNN reaches p_L 0.084 (MWPM parity). At d=9, r=9 on
the same protocol it sits at chance: training accuracy 0.525 after 13 epochs at 10M
shots, against a 0.489 base rate. A 10k-shot overfit test reaches 0.970 training
accuracy, so the architecture can represent the mapping and the fault is not structural
in that sense. Two of the RCNN's inputs are det_bits (grouped into kernels here) and
det_evts. Everything that consumes det_evts alone works at d=9: MWPM reaches 0.122 and
the GRU student reaches 0.328. The det_bits path is the part d=9 stresses that nothing
else does, so it needs an independent check rather than an argument.

What d=9 exercises that d=5 does not. group_det_bits_kxk walks kernel positions
(shift_x, shift_y) over n_shifts = d-k+1 values. get_kernel_parity_flips marks a shift
as boundary when it equals 0 or n_shifts-1 and interior otherwise, and independently
sets flip_x from (shift_x + shift_y) % 2.

  d=5, k=3: n_shifts=3, interior shifts are {1}, so the only interior kernel is (1,1),
            whose flip_x is (1+1) % 2 = 0. Interior kernels are never flipped.
  d=9, k=3: n_shifts=7, interior shifts are {1,2,3,4,5}, giving 25 interior kernels, and
            (shift_x + shift_y) % 2 is 1 for half of them. Flipped interior kernels
            occur only at d >= 7.

So a fault in the flipped-interior branch would leave every d=5 result correct while
corrupting most of the d=9 kernels.

Method. Feed one-hot det_bits: sample i has detector bit i set and all others clear.
Read which (kernel, slot) positions light up. Two properties are then checked against
the geometry, independently of the grouping code's own bookkeeping:

  1  Round alignment. Detector bit i encodes (round, measure_qubit). A kernel slot
     encodes (round, kernel_measure_qubit). The rounds must agree: kernel grouping
     regroups qubits within a round and must never move a bit across rounds.

  2  Window membership. shift_frame maps the kernel's own measure qubits into d-lattice
     coordinates for a given (shift_x, shift_y), applying the flips. The set of d-lattice
     measure qubits that light a kernel must equal the set shift_frame predicts. This is
     the check that fails if the flip is applied in the wrong direction, or if an
     interior kernel reads the wrong window.

  3  Slot injectivity. Each (kernel, slot) must be lit by exactly one detector. A slot lit
     by two detectors means bits are being merged; a slot lit by none means the slot is
     constant zero and carries no information.

Set comparison is used for property 2 so the result does not depend on the order the
grouping code happens to emit slots in. Property 3 covers ordering faults separately.

Run:

  python audit_det_bits_geometry.py              # d=5 and d=9, k=3
  python audit_det_bits_geometry.py --d 9 --r 9  # one configuration
"""
import argparse

import numpy as np

from circuit_partition import (get_measure_qubits_ord, get_kernel_parity_flips,
                               shift_frame, group_det_bits_kxk)


def banner(s):
    print(f"\n{'=' * 72}\n{s}\n{'=' * 72}", flush=True)


def kernel_position_table(d, k):
    """Every kernel position with its boundary/interior parity and its flips.

    Returned so the caller can report how many flipped interior kernels a given distance
    produces, which is the configuration d=5 never reaches.
    """
    n_shifts = d - k + 1
    rows = []
    for shift_y in range(n_shifts):
        for shift_x in range(n_shifts):
            parity_x, parity_y, flip_x, flip_y = get_kernel_parity_flips(
                n_shifts, shift_x, shift_y)
            interior = (parity_x == 0 and parity_y == 0)
            rows.append(dict(idx=shift_x + shift_y * n_shifts,
                             shift_x=shift_x, shift_y=shift_y,
                             parity_x=parity_x, parity_y=parity_y,
                             flip_x=flip_x, flip_y=flip_y, interior=interior))
    return rows


def expected_window_coords(d, k, shift_x, shift_y):
    """d-lattice coordinates the kernel at (shift_x, shift_y) should read.

    Built by pushing the kernel's own measure qubits through shift_frame, which is the
    same function group_det_bits_kxk uses to place a kernel on the lattice. Comparing the
    grouping's actual behaviour against this catches a disagreement between the placement
    and the bit selection.
    """
    measure_kxk = get_measure_qubits_ord(k)
    _, shifted = shift_frame(None, measure_kxk, k, d, shift_x, shift_y)
    return {tuple(q[2]) for q in shifted}


def audit(d, r, k, use_rotated_z=True):
    banner(f"det_bits kernel grouping: d={d}, r={r}, k={k}")

    measure_dxd = sorted(get_measure_qubits_ord(d))
    n_measure = len(measure_dxd)                 # d^2 - 1
    n_shifts = d - k + 1
    n_kernels = n_shifts ** 2
    na = k ** 2 - 1
    n_det = r * n_measure

    rows = kernel_position_table(d, k)
    n_interior = sum(1 for x in rows if x['interior'])
    n_flipped_interior = sum(1 for x in rows if x['interior'] and (x['flip_x'] or x['flip_y']))
    print(f"  detectors per shot: {n_det} = {r} rounds x {n_measure} measure qubits")
    print(f"  kernels: {n_kernels} = {n_shifts}^2,  slots per kernel: {r} x {na}")
    print(f"  interior kernels: {n_interior}   of which flipped: {n_flipped_interior}")
    if n_flipped_interior == 0:
        print("  (this distance never produces a flipped interior kernel)")

    # One-hot probe: sample i carries detector bit i and nothing else. int8 matches the
    # binary_t the training path uses for det_bits.
    probe = np.eye(n_det, dtype=np.int8)
    grouped, _, _, _ = group_det_bits_kxk(
        probe, d, r, k, use_rotated_z, data_bits_dxd=None,
        binary_t=np.int8, idx_t=np.int32, make_translation_map=False)
    grouped = np.asarray(grouped)
    exp_shape = (n_kernels, n_det, r * na)
    print(f"  grouped shape: {grouped.shape}   expected: {exp_shape}")
    if grouped.shape != exp_shape:
        print("  shape mismatch; the remaining checks would be meaningless")
        return False

    ok = True

    # Property 1 and 3: build the detector -> (kernel, slot) map from the probe response.
    # slot_sources[kernel][slot] collects every detector index that lights that slot.
    slot_sources = [dict() for _ in range(n_kernels)]
    for det_idx in range(n_det):
        lit = np.argwhere(grouped[:, det_idx, :] == 1)
        for kern, slot in lit:
            slot_sources[kern].setdefault(int(slot), []).append(det_idx)

    bad_round, multi, empty = [], [], []
    for kern in range(n_kernels):
        for slot in range(r * na):
            srcs = slot_sources[kern].get(slot, [])
            if len(srcs) == 0:
                empty.append((kern, slot))
                continue
            if len(srcs) > 1:
                multi.append((kern, slot, srcs))
            for det_idx in srcs:
                if det_idx // n_measure != slot // na:
                    bad_round.append((kern, slot, det_idx))

    print(f"\n  round alignment : {'ok' if not bad_round else f'{len(bad_round)} bits cross rounds'}")
    print(f"  slot injectivity: {'ok' if not multi else f'{len(multi)} slots lit by >1 detector'}")
    print(f"  slot coverage   : {'ok' if not empty else f'{len(empty)} slots never lit'}")
    if bad_round:
        ok = False
        for kern, slot, det_idx in bad_round[:5]:
            print(f"      kernel {kern} slot {slot} <- detector {det_idx} "
                  f"(round {det_idx // n_measure} into slot round {slot // na})")
    if multi:
        ok = False
        for kern, slot, srcs in multi[:5]:
            print(f"      kernel {kern} slot {slot} <- detectors {srcs}")
    if empty:
        ok = False
        print(f"      first few: {empty[:5]}")

    # Property 2: the set of d-lattice measure qubits feeding each kernel must equal the
    # set shift_frame places there. Reported split by kernel class so a fault confined to
    # flipped interior kernels is visible rather than averaged away.
    print()
    mismatches = []
    for row in rows:
        kern = row['idx']
        got = set()
        for slot, srcs in slot_sources[kern].items():
            for det_idx in srcs:
                got.add(tuple(measure_dxd[det_idx % n_measure][2]))
        want = expected_window_coords(d, k, row['shift_x'], row['shift_y'])
        if got != want:
            mismatches.append((row, got, want))

    kind = lambda rw: ('interior' if rw['interior'] else 'boundary') + \
                      ('+flipped' if (rw['flip_x'] or rw['flip_y']) else '')
    classes = {}
    for row in rows:
        classes.setdefault(kind(row), [0, 0])[0] += 1
    for row, _, _ in mismatches:
        classes[kind(row)][1] += 1
    print(f"  window membership by kernel class:")
    for name, (total, bad) in sorted(classes.items()):
        status = 'ok' if bad == 0 else f'{bad}/{total} MISMATCH'
        print(f"      {name:<18} {total:>3} kernels   {status}")

    if mismatches:
        ok = False
        row, got, want = mismatches[0]
        print(f"\n  first mismatch: kernel {row['idx']} at shift "
              f"({row['shift_x']}, {row['shift_y']}), flips ({row['flip_x']}, {row['flip_y']})")
        print(f"      read but not expected: {sorted(got - want)}")
        print(f"      expected but not read: {sorted(want - got)}")

    print(f"\n  result: {'PASS' if ok else 'FAIL'} for d={d}, r={r}, k={k}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=None, help='single distance to check')
    ap.add_argument('--r', type=int, default=None, help='rounds for that distance')
    ap.add_argument('--k', type=int, default=3)
    args = ap.parse_args()

    if args.d is not None:
        cases = [(args.d, args.r if args.r is not None else args.d)]
    else:
        # d=5 is the control: it is the configuration every published RCNN number came
        # from, so it must pass. d=7 is included because it is the smallest distance that
        # produces a flipped interior kernel, which isolates that branch from d=9's other
        # differences.
        cases = [(5, 5), (7, 7), (9, 9)]

    results = {}
    for d, r in cases:
        results[(d, r)] = audit(d, r, args.k)

    banner("summary")
    for (d, r), ok in results.items():
        print(f"  d={d}, r={r}: {'PASS' if ok else 'FAIL'}")
    if all(results.values()):
        print("\n  The det_bits kernel grouping is internally consistent at every distance")
        print("  checked: bits stay within their round, every kernel slot is fed by exactly")
        print("  one detector, and the lattice sites feeding each kernel are the ones")
        print("  shift_frame places there.")
        print()
        print("  Scope. This compares group_det_bits_kxk against shift_frame, and the two")
        print("  share that placement function, so a fault living inside shift_frame itself")
        print("  would satisfy both sides and pass here. What this rules out is a")
        print("  disagreement between kernel placement and bit selection, not an error in")
        print("  the shared convention. An independent check would need the expected")
        print("  windows derived from the lattice definition instead.")
    else:
        print("\n  A failing distance here would corrupt the RCNN's det_bits input and")
        print("  would need fixing before any d=9 result is trusted.")


if __name__ == '__main__':
    main()
