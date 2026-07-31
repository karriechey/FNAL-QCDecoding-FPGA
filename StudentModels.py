#!/usr/bin/env python3
# Created: 2026-07-29
# Last modified: 2026-07-31
"""hls4ml-native student architectures for distillation from the custom RCNN teacher.

Why train a student model?
The teacher (CNNModel.FullRCNNModel) cannot be parsed by hls4ml's automated QKeras flow:
its state-combiner math uses pow/sqrt/log/gather ops that sit outside hls4ml's supported
layer set. That is the synthesis blocker, and no amount of quantizing the teacher
removes it. A student built only from layers hls4ml supports is synthesizable by
construction, and the combiner math is never implemented in HLS at all -- it survives
only as the supervision signal that shaped the student's weights.

The two students
Both consume the same per-shot syndrome information as the teacher and emit one score.
  mlp -- flatten the syndrome and push it through QDense/ReLU layers. The simplest thing
         that could work, and the cheapest to synthesize. Serves as the floor: if the GRU
         cannot beat it, the recurrence is not earning its resource cost.
  gru -- treat the syndrome as a sequence over detector timesteps. Structurally matches
         how the error information arrives in a real decoder and what the teacher's
         round-recurrent structure encodes. hls4ml supports recurrent layers (QGRU parses).

Input layout (d=5, r=3)
The pool stores two per-shot arrays, each 72 wide:
  det_bits -- raw stabilizer measurement outcomes, 3 rounds x 24 stabilizers
  det_evts -- detector events, the round-to-round XOR of those outcomes

These two have DIFFERENT layouts. det_evts is not 3x24: its 72 detectors span 4
timesteps over 24 plaquette positions with unequal occupancy (12/24/24/12), because the
first round detects only Z stabilizers and the last is derived from the data-qubit
measurements. detector_sequence_layout() reads this off the stim circuit; the MLP
flattens so it is unaffected, while the GRU consumes the scattered [4, 24] form.
det_evts is what MWPM consumes and what the FPGA NN-decoder literature feeds its networks,
so it is the default student input and the smallest one to route on-chip. The teacher sees
BOTH arrays, so `--inputs evts+bits` exists to test whether that extra channel is carrying
information the student needs; keeping it a flag makes "does the raw measurement channel
matter" a measured result rather than an assumption baked into the architecture.

Output is a LOGIT （log-it), not a probability
The final layer is linear and the sigmoid is NOT part of the model. Two reasons:
 1. Distillation is defined in logit space (the temperature divides a logit), so a model
    that natively emits logits needs no numerically lossy inversion of its own sigmoid.
 2. sigmoid is monotonic, so thresholding the logit at 0 gives exactly the same decision
    as thresholding the probability at 0.5. On FPGA the sigmoid is therefore pure cost
    with zero effect on the decode, and dropping it saves a LUT/table.
Convert with sigmoid() only when a calibrated probability is actually wanted.

Quantization hooks:
weight_bits / act_bits default to None, which builds plain float Keras layers -- that is
the Phase-2 distillation setting, where the only question is how well the student can
imitate the teacher at full precision. Passing bit-widths swaps in the QKeras equivalents
so the Phase-3 sweep reuses this exact file rather than a divergent copy of the
architecture. The quantizer convention matches the teacher sweep: quantized_bits(B, 1),
i.e. ap_fixed<B,2> = 1 sign + 1 integer + (B-2) fractional bits.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras import Model


def _patch_qkeras_recurrent_nest():
    """Restore tf.python.util.nest.is_sequence for QKeras 0.9's recurrent cells.

    QKeras 0.9's QGRUCell/QLSTMCell call `nest.is_sequence(states)`. TensorFlow renamed
    that helper to `nest.is_nested` and deleted the old alias, so on the repo's pinned
    TF 2.15 every QGRU/QLSTM forward pass dies with

        AttributeError: module 'tensorflow.python.util.nest' has no attribute 'is_sequence'

    The two functions are the same predicate under different names, so aliasing the old
    name back onto the module is a faithful fix, not a workaround that changes behaviour.
    This is the same class of QKeras-vs-TF regression as the convert_to_npdtype one and
    is likewise a candidate upstream patch.

    Only installed if the attribute is genuinely absent, so a future QKeras or TF that
    provides its own is left untouched.
    """
    try:
        from tensorflow.python.util import nest as _nest
    except ImportError:  # pragma: no cover - TF internals moved; let QKeras fail loudly
        return
    if not hasattr(_nest, 'is_sequence') and hasattr(_nest, 'is_nested'):
        _nest.is_sequence = _nest.is_nested


_patch_qkeras_recurrent_nest()


_LAYOUT_CACHE = {}


def detector_sequence_layout(d, rounds, p):
    """Map the flat det_evts vector onto (timestep, plaquette position).

    For rotated_memory_z at d=5, r=3 the 72 detectors are NOT 3 rounds x 24 stabilizers.
    Reading the coordinates off the stim circuit gives 4 timesteps over 24 distinct (x,y)
    plaquette positions, with unequal occupancy:

        t=0: 12 detectors    (first round: only the Z stabilizers detect)
        t=1: 24
        t=2: 24
        t=3: 12             (final round, derived from the data-qubit measurements)

    The short timesteps' positions are subsets of the full 24, so every timestep can be
    written into a fixed 24-wide slot vector with zeros where nothing was measured. That
    keeps column j meaning the same plaquette at every timestep, which is what makes the
    recurrence meaningful, and keeps the input a fixed rectangular shape for hls4ml.

    Returns (n_timesteps, n_positions, index_map) where index_map[t, j] is the detector
    index for position j at timestep t, or -1 if that position is not measured then.
    """
    key = (d, rounds, round(p, 6))
    if key in _LAYOUT_CACHE:
        return _LAYOUT_CACHE[key]

    from circuit_generators import get_builtin_circuit
    circ = get_builtin_circuit(
        'surface_code:rotated_memory_z', distance=d, rounds=rounds,
        before_round_data_depolarization=p, after_reset_flip_probability=p,
        after_clifford_depolarization=p, before_measure_flip_probability=p)
    coords = circ.get_detector_coordinates()
    n_det = circ.num_detectors

    xy = [(coords[i][0], coords[i][1]) for i in range(n_det)]
    ts = [int(coords[i][2]) for i in range(n_det)]
    positions = sorted(set(xy))
    pos_index = {q: j for j, q in enumerate(positions)}
    n_t, n_pos = max(ts) + 1, len(positions)

    index_map = np.full((n_t, n_pos), -1, dtype=np.int32)
    for i in range(n_det):
        index_map[ts[i], pos_index[xy[i]]] = i

    _LAYOUT_CACHE[key] = (n_t, n_pos, index_map)
    return _LAYOUT_CACHE[key]


def to_sequence(det_evts, d, rounds, p):
    """Scatter flat det_evts [N, n_det] into [N, n_timesteps, n_positions], zero-filled.

    Done in numpy rather than as a Lambda/gather inside the model: the reordering is a
    fixed wiring of the input buffer, so on FPGA it costs nothing, and hls4ml never has to
    parse a gather.
    """
    n_t, n_pos, index_map = detector_sequence_layout(d, rounds, p)
    ev = np.asarray(det_evts)
    out = np.zeros((ev.shape[0], n_t, n_pos), dtype=np.float32)
    for t in range(n_t):
        cols = index_map[t]
        present = cols >= 0
        out[:, t, present] = ev[:, cols[present]]
    return out


def input_width(d, rounds, inputs):
    """Per-shot feature width for a given input choice. r*(d^2-1) per array."""
    per_array = rounds * (d ** 2 - 1)
    if inputs == 'evts':
        return per_array
    if inputs == 'evts+bits':
        return 2 * per_array
    raise ValueError(f"unknown inputs={inputs!r} (expected 'evts' or 'evts+bits')")


def assemble_features(det_bits, det_evts, inputs, student='mlp', d=5, rounds=3, p=0.010):
    """Build the student's input array from the pool arrays, as float32.

    Kept here rather than in the trainer so the model definition and the data layout it
    expects can never drift apart. Cast to float32 because the int8 pool arrays would
    otherwise be cast implicitly at every batch.

    student='gru' returns [N, n_timesteps, n_positions]; anything else returns the flat
    [N, n_features] vector, where feature order is irrelevant to a Dense stack.
    """
    if student == 'gru':
        if inputs != 'evts':
            raise ValueError("the GRU student supports inputs='evts' only")
        return to_sequence(det_evts, d, rounds, p)

    if inputs == 'evts':
        x = np.asarray(det_evts)
    elif inputs == 'evts+bits':
        x = np.concatenate([np.asarray(det_evts), np.asarray(det_bits)], axis=1)
    else:
        raise ValueError(f"unknown inputs={inputs!r}")
    return x.astype(np.float32)


def _quantizers(weight_bits):
    """QKeras quantizer objects for a weight word length, or None for float layers.

    quantized_bits(B, 1, alpha=1) == ap_fixed<B,2>: 1 sign bit, 1 integer bit, B-2
    fractional bits. alpha=1 pins the scale to 1 so the fixed-point interpretation is the
    literal one hls4ml will synthesize -- QKeras's default per-channel alpha would fold a
    learned float scale in front of the weights, which is not what the FPGA implements.
    Integer split is pinned a priori here exactly as in the teacher sweep; Phase 4 range
    profiling revisits it with measured tensor ranges.
    """
    if weight_bits is None or weight_bits >= 32:
        return None
    from qkeras import quantized_bits
    return quantized_bits(weight_bits, 1, alpha=1)


def _dense(units, name, weight_bits):
    """A Dense layer, quantized if a weight word length was given."""
    q = _quantizers(weight_bits)
    if q is None:
        return KL.Dense(units, name=name)
    from qkeras import QDense
    return QDense(units, kernel_quantizer=q, bias_quantizer=q, name=name)


def _relu(name, act_bits):
    """A ReLU activation, quantized if an activation word length was given.

    quantized_relu(B, I) is unsigned (ReLU output is non-negative, so no sign bit is
    spent) with I integer bits. I is pinned to 2 here as a starting point; Phase 4
    profiling replaces it with the measured per-tensor value.
    """
    if act_bits is None or act_bits >= 32:
        return KL.Activation('relu', name=name)
    from qkeras import QActivation
    return QActivation(f'quantized_relu({act_bits},2)', name=name)


def build_mlp_student(d=5, rounds=3, inputs='evts', hidden=(128, 128),
                      weight_bits=None, act_bits=None, name='mlp_student'):
    """Flatten-and-stack student. Returns a Keras Model emitting ONE LOGIT per shot."""
    n_feat = input_width(d, rounds, inputs)
    x_in = KL.Input(shape=(n_feat,), name='syndrome')
    h = x_in
    for i, units in enumerate(hidden):
        h = _dense(units, f'dense_{i}', weight_bits)(h)
        h = _relu(f'relu_{i}', act_bits)(h)
    # Linear head: the output is a logit. See module docstring for why no sigmoid.
    out = _dense(1, 'logit', weight_bits)(h)
    return Model(x_in, out, name=name)


def build_gru_student(d=5, rounds=3, inputs='evts', units=64, hidden=(), p=0.010,
                      weight_bits=None, act_bits=None, name='gru_student'):
    """Recurrent student: one GRU timestep per detector round.

    Input is [n_timesteps, n_positions] as produced by to_sequence() -- for d=5, r=3 that
    is (4, 24). The scatter happens in the data pipeline, so this model takes the sequence
    directly and contains no reshape or gather.

    `hidden` optionally adds Dense layers between the GRU and the head; empty by default.

    inputs='evts+bits' is not supported here: det_bits is 3 rounds x 24 raw stabilizer
    measurements, a different time base from det_evts' 4 detector timesteps, and there is
    no correct way to align them without a decision about which to resample.
    """
    if inputs != 'evts':
        raise ValueError("the GRU student supports inputs='evts' only "
                         f"(got {inputs!r}); det_bits has a different time base")
    n_t, n_pos, _ = detector_sequence_layout(d, rounds, p)
    seq = KL.Input(shape=(n_t, n_pos), name='syndrome_sequence')
    x_in = seq

    # reset_after pinned explicitly: Keras GRU defaults to True, QKeras QGRU to False, so
    # leaving it implicit makes the float and quantized students structurally different
    # layers (differing bias shape) and their results incomparable.
    if weight_bits is None or weight_bits >= 32:
        h = KL.GRU(units, return_sequences=False, reset_after=False, name='gru')(seq)
    else:
        from qkeras.qrecurrent import QGRU
        q = _quantizers(weight_bits)
        # State/activation quantizers are given the activation word length when one was
        # requested; at Phase 2 (act_bits=None) the recurrent activations stay float and
        # only the weights are quantized, matching the teacher sweep's weights-only stage.
        a_tanh = f'quantized_tanh({act_bits})' if act_bits else 'tanh'
        a_sigm = f'quantized_sigmoid({act_bits})' if act_bits else 'sigmoid'
        state_q = _quantizers(act_bits) if act_bits else _quantizers(weight_bits)
        h = QGRU(units, return_sequences=False, reset_after=False,
                 kernel_quantizer=q, recurrent_quantizer=q, bias_quantizer=q,
                 state_quantizer=state_q,
                 activation=a_tanh, recurrent_activation=a_sigm,
                 name='gru')(seq)

    for i, n in enumerate(hidden):
        h = _dense(n, f'dense_{i}', weight_bits)(h)
        h = _relu(f'relu_{i}', act_bits)(h)
    out = _dense(1, 'logit', weight_bits)(h)
    return Model(x_in, out, name=name)


def build_student(student, d=5, rounds=3, inputs='evts', hidden=(128, 128), units=64,
                  weight_bits=None, act_bits=None):
    """Dispatch on the student name so trainers take a plain string argument."""
    if student == 'mlp':
        return build_mlp_student(d=d, rounds=rounds, inputs=inputs, hidden=hidden,
                                 weight_bits=weight_bits, act_bits=act_bits)
    if student == 'gru':
        return build_gru_student(d=d, rounds=rounds, inputs=inputs, units=units,
                                 hidden=hidden if hidden else (),
                                 weight_bits=weight_bits, act_bits=act_bits)
    raise ValueError(f"unknown student={student!r} (expected 'mlp' or 'gru')")


def logits_to_prob(logits):
    """sigmoid, for when a calibrated probability is wanted instead of a decision."""
    return tf.sigmoid(logits).numpy() if tf.is_tensor(logits) else 1.0 / (
        1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))


def student_pred_and_correct(logits, truth):
    """Decision convention for the student, mirroring eval_on_tail.rcnn_pred_and_correct.

    The student emits a logit, so the threshold is 0, which is exactly equivalent to
    thresholding sigmoid(logit) at 0.5. Single source of this convention so no downstream
    script reimplements it with a different threshold.
    """
    pred = (np.asarray(logits).reshape(-1) > 0.0).astype(np.int8)
    truth = np.asarray(truth).reshape(-1).astype(np.int8)
    return pred, (pred == truth)
