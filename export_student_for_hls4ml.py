#!/usr/bin/env python3
# Created: 2026-08-04
# Last modified: 2026-08-04
"""Export a student as a FULL Keras model (.h5), uploadable to hls4ml front-ends.

train_student.py calls model.save_weights(), which writes weight values with no graph.
hls4ml needs the architecture -- reading the layer structure is the entire job -- so a
.weights.h5 cannot be uploaded to KalEdge or passed to hls4ml.convert_from_keras_model.
This writes model.save(), which stores architecture + weights together.

Every export is verified by reloading the written file and comparing predictions against
the in-memory model on random input -- a file that cannot round-trip locally will not
parse remotely, and catching that here costs a second.

  python export_student_for_hls4ml.py --out-dir ~/Documents/FNAL-QCDecoding-FPGA/hls4ml_upload
"""
import argparse
import os
import numpy as np


def verify_roundtrip(model, path, x):
    """Reload the saved file and confirm it reproduces the in-memory model's output.

    Guards against a save that silently drops or mangles a layer. Returns the max
    absolute prediction difference, which should be exactly 0 for a clean FP32 save.
    """
    import tensorflow as tf
    reloaded = tf.keras.models.load_model(path, compile=False)
    a = model.predict(x, verbose=0).reshape(-1)
    b = reloaded.predict(x, verbose=0).reshape(-1)
    return float(np.abs(a - b).max()), reloaded


def describe(model):
    """One line per layer: name, type, output shape, param count. Printed so the exported
    graph is on the record next to the parse result -- when hls4ml rejects a layer, the
    log should already say which layers were present.
    """
    lines = []
    for lyr in model.layers:
        try:
            shape = str(lyr.output_shape)
        except AttributeError:
            shape = '?'
        lines.append(f"    {lyr.name:22s} {type(lyr).__name__:18s} {shape:18s} "
                     f"{lyr.count_params():>8,}")
    return '\n'.join(lines)


def export_one(build_kwargs, tag, out_dir, weights=None, reset_after=None):
    """Build one student, optionally load weights, save the full model, verify, describe."""
    import tensorflow as tf
    from StudentModels import build_student, detector_sequence_layout

    student = build_kwargs['student']
    # build_student() does not forward `p` to the GRU builder (it takes StudentModels'
    # own default, 0.010, which is this study's value). Call the GRU builder directly so
    # the detector layout is derived from the p we were actually asked for, rather than
    # relying on two defaults happening to agree.
    if student == 'gru':
        from StudentModels import build_gru_student
        kw = {k: v for k, v in build_kwargs.items() if k != 'student'}
        model = build_gru_student(**kw)
    else:
        kw = {k: v for k, v in build_kwargs.items() if k != 'p'}
        model = build_student(**kw)

    # reset_after override: rebuild the GRU layer's config through the saved-config path
    # rather than mutating the layer in place, so the exported graph is one Keras itself
    # produced and there is no half-applied state.
    if reset_after is not None and student == 'gru':
        cfg = model.get_config()
        for lyr in cfg['layers']:
            if lyr['class_name'] == 'GRU':
                lyr['config']['reset_after'] = reset_after
        model = tf.keras.Model.from_config(cfg)

    if weights:
        if not os.path.exists(weights):
            raise SystemExit(f"[export] MISSING weights {weights}")
        model.load_weights(weights)
        provenance = os.path.basename(weights)
    else:
        # Untrained weights. Fine for a parse test (see module docstring), and stated
        # plainly so this artifact is never mistaken for one carrying real numbers.
        provenance = 'UNTRAINED (structural parse test only)'

    # Input for the round-trip check, shaped like the model expects.
    if student == 'gru':
        n_t, n_pos, _ = detector_sequence_layout(build_kwargs.get('d', 5),
                                                 build_kwargs.get('rounds', 3),
                                                 build_kwargs.get('p', 0.010))
        x = np.random.randint(0, 2, (8, n_t, n_pos)).astype(np.float32)
    else:
        n_feat = model.input_shape[-1]
        x = np.random.randint(0, 2, (8, n_feat)).astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{tag}.h5')
    # save_format='h5' explicitly: the full-model HDF5 format hls4ml front-ends accept.
    model.save(path, save_format='h5')

    max_diff, reloaded = verify_roundtrip(model, path, x)
    size_kb = os.path.getsize(path) / 1024

    print(f"  {tag}")
    print(f"    file        {path}  ({size_kb:.1f} KB)")
    print(f"    weights     {provenance}")
    print(f"    input       {model.input_shape}   output {model.output_shape}")
    print(f"    params      {model.count_params():,}")
    print(f"    round-trip  max|pred diff| = {max_diff:.3e}"
          f"{'  OK' if max_diff == 0.0 else '  <-- NONZERO, inspect before uploading'}")
    print(describe(model))
    print()
    return path, max_diff


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir',
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'hls4ml_upload'),
                    help='where to write the uploadable .h5 files')
    ap.add_argument('--d', type=int, default=5)
    ap.add_argument('--p', type=float, default=0.010)
    ap.add_argument('--rounds', type=int, default=3)
    # Ladder geometry: the 76-run pools_t200k study used MLP hidden=(128,128) -> 25,985
    # params and GRU units=64 -> 17,153 params. Defaults reproduce those exact graphs.
    ap.add_argument('--mlp-hidden', type=int, nargs='*', default=[128, 128])
    ap.add_argument('--gru-units', type=int, default=64)
    ap.add_argument('--students', nargs='*', default=['mlp', 'gru'])
    ap.add_argument('--mlp-weights', default=None,
                    help='optional .weights.h5 for the MLP (not needed for a parse test)')
    ap.add_argument('--gru-weights', default=None,
                    help='optional .weights.h5 for the GRU (not needed for a parse test)')
    ap.add_argument('--both-reset-after', action='store_true', default=True,
                    help='export the GRU at reset_after False AND True, isolating it as '
                         'the cause if the GRU fails to parse (default on)')
    ap.add_argument('--single-reset-after', dest='both_reset_after', action='store_false')
    ap.add_argument('--cpu', action='store_true', default=True)
    args = ap.parse_args()

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')
    assert tf.__version__.startswith('2.15'), (
        f'need TF 2.15.x (Keras 2); got TF {tf.__version__}.')

    print(f"[export] full-model .h5 for hls4ml upload -> {args.out_dir}")
    print(f"[export] TF {tf.__version__}\n")

    written = []
    if 'mlp' in args.students:
        written.append(export_one(
            dict(student='mlp', d=args.d, rounds=args.rounds, inputs='evts',
                 hidden=tuple(args.mlp_hidden)),
            'student_mlp_evts_fp32', args.out_dir, weights=args.mlp_weights))

    if 'gru' in args.students:
        base = dict(student='gru', d=args.d, rounds=args.rounds, inputs='evts',
                    units=args.gru_units, hidden=(), p=args.p)
        # reset_after=False is what StudentModels pins and what the quantized student
        # will be, so it is the primary artifact.
        written.append(export_one(base, 'student_gru_evts_fp32_reset_after_False',
                                  args.out_dir, weights=args.gru_weights,
                                  reset_after=False))
        if args.both_reset_after:
            # The Keras default, and what hls4ml's GRU parser conventionally expects.
            # Only a diagnostic: if this one parses and the other does not, reset_after
            # is the sole cause.
            written.append(export_one(base, 'student_gru_evts_fp32_reset_after_True',
                                      args.out_dir, weights=None, reset_after=True))

    print("[export] done. Upload these to the hls4ml tab's "
          "'Upload Pre-Trained Model' field:")
    for path, diff in written:
        print(f"    {path}   (round-trip {diff:.0e})")
    print("\n[export] These carry UNTRAINED weights unless --*-weights was passed. That is "
          "correct for a structural parse test -- the graph is what is under test -- but "
          "do NOT read any accuracy or numerical result off them.")


if __name__ == '__main__':
    run()
