"""
crumble_polygons.py

Add surface-code stabilizer polygons to a Stim rotated-surface-code circuit's
Crumble URL, placed in the first (bottom) layer so Crumble draws them as a
persistent background beneath the qubits/gates across *every* tick.

Why layer 0: Crumble (glue/crumble/draw/main_draw.js) draws polygons before the
qubit grid and gates, and for the current tick it uses the most recent
polygon-bearing layer <= current. Declaring the full tiling once, before the
first TICK, makes it show on all ticks at the bottom of the z-stack.

Colors follow the detslice convention: Z = blue, X = red.

Usage in failure_case_explorer.ipynb (cell 12):

    from crumble_polygons import to_crumble_url_with_code
    url = to_crumble_url_with_code(CIRC, mark={1: expl})

and in the browse loop (cell 18):

    u = to_crumble_url_with_code(CIRC, mark={1: ex})
"""
import math


def stabilizer_polygons(circuit):
    """[( (r,g,b,a), [ordered qubit ids] ), ...] for each stabilizer plaquette."""
    coords = {q: tuple(v) for q, v in circuit.get_final_qubit_coordinates().items()}

    def tset(inst):
        return [g.value for g in inst.targets_copy() if g.is_qubit_target]

    data, anc, hgate = set(), set(), set()
    for inst in circuit.flattened():
        if inst.name == 'M':
            data |= set(tset(inst))          # final data readout
        elif inst.name == 'MR':
            anc |= set(tset(inst))           # per-round syndrome ancillas
        elif inst.name == 'H':
            hgate |= set(tset(inst))         # ancillas rotated into X basis
    x_anc = anc & hgate                      # X stabilizers
    z_anc = anc - hgate                      # Z stabilizers

    Z_RGBA = (0, 0, 1, 0.25)                 # blue = Z
    X_RGBA = (1, 0, 0, 0.25)                 # red  = X

    def plaquette(a):
        ax, ay = coords[a]
        nbrs = [q for q in data
                if abs(coords[q][0] - ax) == 1 and abs(coords[q][1] - ay) == 1]
        nbrs.sort(key=lambda q: math.atan2(coords[q][1] - ay, coords[q][0] - ax))
        return nbrs

    return ([(Z_RGBA, plaquette(a)) for a in sorted(z_anc)]
            + [(X_RGBA, plaquette(a)) for a in sorted(x_anc)])


def to_crumble_url_with_code(circuit, mark=None):
    """Same as circuit.to_crumble_url(mark=...) but with the stabilizer tiling
    injected into layer 0 so it renders as a persistent bottom-layer background."""
    poly_ops = ';'.join(
        f"POLYGON({r},{g},{b},{a})" + '_'.join(str(q) for q in ids)
        for (r, g, b, a), ids in stabilizer_polygons(circuit)
    )
    url = circuit.to_crumble_url(mark=mark) if mark else circuit.to_crumble_url()
    head, body = url.split('#circuit=')
    toks = body.split(';')
    k = 0
    while k < len(toks) and toks[k].startswith('Q('):   # after the qubit-coord decls
        k += 1
    return head + '#circuit=' + ';'.join(toks[:k] + [poly_ops] + toks[k:])
