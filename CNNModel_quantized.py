# Created: 2026-07-12
# Last modified: 2026-07-20
"""Weight quantization for the reference architecture's FullRCNNModel -- modified version of CNNModel.py.

Note
-------------------
Standard QKeras layer-substitution (Dense -> QDense) cannot reach the reference architecture's
hand-managed `add_weight` tensors, which live inside custom `Layer.call()` math.
This code instead applies `quantized_bits` directly to those weight tensors at their
point of use. `FullRCNNModel` hard-codes which child layers it constructs, so a
plain subclass cannot inject quantization into the nested kernels. This module
therefore installs quantized *point-of-use wrappers* onto the custom layer classes
(per class, mirroring each original method, adding only `WeightQuant.q(...)` at the
weight site).

USAGE
-----
    from CNNModel_quantized import build_quantized_rcnn
    model = build_quantized_rcnn(
        weight_bits=8,                       # None or >=32 => FP32 no-op (baseline)
        obs_type='ZL', code_distance=5, kernel_distance=3, rounds=3,
        hidden_specs=[100, 100], npol=2, stop_round=None,
        has_nonuniform_response=False, do_all_data_qubits=False, return_all_rounds=False,
    )
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

Equivalently: call `enable_weight_quantization(bits)` before constructing a plain
`CNNModel.FullRCNNModel(...)`.

FIXED-POINT LAYOUT (matches QUANTIZATION_SCHEME.md)
--------------------------------------------------
`quantized_bits(B, 1)` with keep_negative default True => 1 sign + 1 integer +
(B-2) fractional bits, range ~[-2, +2). Weights only; activations left full precision
(the brief's Experiment-1 sweep varies weight bit-width only).

CAVEAT (process-global, one config per run)
-------------------------------------------
The patches are installed on the shared classes and read `WeightQuant._quantizer`
at forward time -- this mirrors the codebase's own static setters
(RCNNInitialStateKernel.set_rounds_first, CNNStateCorrelator.set_disable_fractions).
Build/train/eval ONE weight_bits config per process, which is exactly how the sweep
driver runs. The patches are inert when bits is None (byte-identical FP32).

WHAT IS QUANTIZED (the r=3 default forward graph)
-------------------------------------------------
- CNNKernelWithEmbedding: kernel_weights_det_bits, kernel_weights_det_evts, kernel_bias
- CNNStateCorrelator:     params_state_evolutions, params_b
- RCNNKernelCombiner:     TranslationFrac_*, TranslationPhase_*, Inverter_*, NonUniformResponseAdj
- StateDecoder:           stock Dense -> weight-quantized QDense
Left unquantized (per brief): RCNNRecurrenceBaseKernel.kernel_weights_triplet_states
(dormant at r=3), and the detector-state embedders.
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense
from qkeras import quantized_bits, QDense

import CNNModel
from CNNModel import (
    FullRCNNModel,
    CNNKernelWithEmbedding,
    CNNStateCorrelator,
    RCNNKernelCombiner,
    StateDecoder,
    DetectorBitStateEmbedder,
    DetectorEventStateEmbedder,
    TripletStateProbEmbedder,
    VariableBounds,
    arrayops_shape,
)


class WeightQuant:
    """Process-global weight-quantization config. `bits` None or >=32 => FP32 no-op."""
    _bits = None
    _quantizer = None

    @staticmethod
    def set_bits(bits):
        if bits is None or bits >= 32:
            WeightQuant._bits = None
            WeightQuant._quantizer = None
        else:
            WeightQuant._bits = bits
            WeightQuant._quantizer = quantized_bits(bits, 1) #arg1 = total bit-width, arg2 = 1 integer bit

    @staticmethod
    def q(w):
        """Quantize a weight tensor at its point of use (no-op in FP32 mode)."""
        if w is None or WeightQuant._quantizer is None:
            return w
        return WeightQuant._quantizer(w)

    @staticmethod
    def make_dense(units, activation=None):
        """Dense in FP32 mode; weight-quantized QDense otherwise."""
        if WeightQuant._quantizer is None:
            return Dense(units, activation=activation)
        q = quantized_bits(WeightQuant._bits, 1)
        return QDense(units, activation=activation,
                      kernel_quantizer=q, bias_quantizer=q)


from qkeras import quantized_relu


class ActQuant:
    """Process-global ACTIVATION-quantization config (Phase 2a). One activation word length
    `bits` (B), swept; integer bits I fixed PER CLASS from the fixed_point_format_table
    (collate_profile.py) / taxonomy. `bits` None or >=32 => qa() is a byte-exact no-op, so
    the model reproduces the w6/act-FP32 anchor exactly (the identity check).

    Integer-bit convention: these are the QKeras `quantized_bits(bits, integer, keep_negative)`
    `integer` argument, which EXCLUDES the sign bit. A signed class with integer=I represents the
    range [-2^I, 2^I); i.e. it corresponds to ap_fixed<bits, I+1> in the paper's sign-inclusive
    notation. (So the weight quantizer's quantized_bits(B,1) == ap_fixed<B,2>, matching the header.)

    Per-class (QKeras integer bits, keep_negative, is_relu):
      zlike  signed   integer=4  post-clip z'' with |z''| <= 12; combiner output + decoder input.
      pf     unsigned integer=0  sigmoid fractions in (0,1); triplet-prob embedder output.
      cphi   signed   integer=0  c_phi = tanh in (-1,1), range [-1,1). We quantize c_phi at its
                                 natural range here; the phase term 2*c_phi is then formed by a
                                 lossless power-of-2 multiply AFTER quantization (see combiner
                                 site). Quantizing the pre-doubled c_phi (rather than 2*c_phi at
                                 integer=1) matches the reference architecture's variable and wastes no fractional bit.
      embed  signed   integer=2  Detector{Bit,Event} embedder OUTPUT. This is NOT a bounded
                                 c_phi/alpha tensor: embed_pol_state returns a (-1,1) diagonal part
                                 plus an UNBOUNDED non-diagonal polynomial, so its width is PROFILED.
                                 integer=2 covers the DetectorEvent max; DetectorBit (which needs
                                 only integer=1) also fits.
      relu   unsigned integer=6  decoder hidden-ReLU outputs. Seeded at the absolute-maximum width
                                 (integer=6), which is SAFE (nothing saturates), so a failed first
                                 sweep cannot be blamed on clipping. Tighten to the 99.9th-percentile
                                 width (integer=4/5) only after confirming the model trains; see
                                 docs/RUN_LOG.md.
    x-like intermediates (kernel/correlator outputs, combiner pre-log products) are NOT quantized
    here -- they span ~10 decades and are un-representable in fixed point; they are handled in
    Phase 4 (log-domain / LSE rewrite).
    """
    _bits = None
    _CLASSES = {              # class -> (QKeras integer bits, keep_negative, is_relu)
        'zlike': (4, True, False),
        'pf':    (0, False, False),
        'cphi':  (0, True, False),   # c_phi in (-1,1); 2*c_phi formed downstream by a power-of-2 shift
        'embed': (2, True, False),
        'relu':  (6, False, True),
    }
    _quantizers = {}

    @staticmethod
    def _fractional_width(bits, I, kneg, is_relu):
        """Fractional bits left after the sign and integer fields are taken out of a B-bit word.

        quantized_relu(B, I) is unsigned: frac = B - I. quantized_bits(B, I, keep_negative) spends
        one bit on the sign when keep_negative: frac = B - I - 1. A class needs frac >= 1 to carry
        ANY sub-integer resolution; frac <= 0 means the representable values are spaced 2^-frac
        apart (frac=-2 => steps of 4), which silently destroys the tensor.
        """
        return bits - I - (1 if (kneg and not is_relu) else 0)

    # Fractional-width thresholds, calibrated against observed Phase-2a runs rather than assumed:
    #   frac <  0  FATAL. Grid spacing is 2^-frac >= 2, coarser than the unit. Observed at B=4
    #              (zlike frac=-1, relu frac=-2): all three seeds collapsed to a constant predictor.
    #   frac == 0  ALLOWED, warned. Grid spacing is exactly 1.0 -- coarse, but still a valid integer
    #              grid. Observed at B=6 (relu frac=0): trained normally and cost only ~+0.0015 p_L
    #              vs its own control. So this is NOT a failure condition; refusing it would reject
    #              a configuration that demonstrably works.
    _FRAC_FATAL_BELOW = 0

    @staticmethod
    def set_bits(bits):
        """Install the per-class activation quantizers for word length `bits`.

        Guarded: QKeras accepts quantized_bits/quantized_relu with a NEGATIVE fractional width
        without raising, and silently returns a quantizer whose representable grid is coarser than
        1.0. That is not a low-precision model -- it is a destroyed one. It happened for real in the
        Phase-2a B=4 runs (2026-07-19): zlike has I=4 signed => frac = 4-4-1 = -1, and the decoder
        ReLU has I=6 unsigned => frac = 4-6 = -2 (values spaced 4 apart over [0,64), so every hidden
        activation below 2.0 rounded to zero). All three seeds trained for hours, never moved off
        val_loss 0.5935, and landed at p_L = 0.2826 = exactly the tail's base rate -- a constant
        predictor. That result is a fixed-point FORMAT artifact, not a statement about whether the
        decoder tolerates 4-bit activations, and reporting it as the latter would be wrong.

        So: refuse to build an infeasible configuration, and say which class failed and what the
        minimum viable B is, rather than burning another multi-hour run to produce garbage. The
        binding constraint is the widest integer field: B_min = max over classes of (I + sign).
        Lowering it requires retuning the integer widths (see set_relu_integer for the ReLU p99.9
        path); zlike's I=4 comes from the architecture's +/-12 clip and is not freely reducible.

        The cutoff is frac < 0, not frac < 1: B=6 leaves the decoder ReLU with frac=0 (resolution
        1.0) and still trained normally, so a zero fractional width is warned about, not rejected.
        """
        ActQuant._bits = None if (bits is None or bits >= 32) else bits
        ActQuant._quantizers = {}
        if ActQuant._bits is None:
            return

        # Validate every class BEFORE constructing any quantizer, so the error names all offenders.
        bad, tight = [], []
        for cls, (I, kneg, is_relu) in ActQuant._CLASSES.items():
            frac = ActQuant._fractional_width(ActQuant._bits, I, kneg, is_relu)
            sign = 1 if (kneg and not is_relu) else 0
            min_B = I + sign            # smallest B giving this class frac >= 0
            if frac < ActQuant._FRAC_FATAL_BELOW:
                bad.append((cls, I, sign, frac, min_B))
            elif frac == 0:
                tight.append(cls)
        if bad:
            b_min = max(m for (_, _, _, _, m) in bad)
            detail = '; '.join(
                f'{cls} (integer={I}, sign={sign}) has fractional width {frac} '
                f'-- needs B >= {m}'
                for (cls, I, sign, frac, m) in bad)
            raise ValueError(
                f'activation word length B={ActQuant._bits} is infeasible for '
                f'{len(bad)} class(es): {detail}. Minimum viable B for this integer-width '
                f'policy is {b_min}. A negative fractional width means the representable values '
                f'are spaced more than 1.0 apart, which drives the model to a constant predictor '
                f'(observed at B=4: p_L = 0.2826 = the tail base rate, on all three seeds). '
                f'To go below B={b_min}, retune the per-class integer widths first '
                f'(e.g. ActQuant.set_relu_integer(4) for the p99.9 ReLU width).')
        if tight:
            print(f'[actquant] WARNING: at B={ActQuant._bits} these classes have ZERO fractional '
                  f'bits (resolution 1.0, integer-valued grid): {", ".join(sorted(tight))}. '
                  f'Representable but coarse -- B=6 ran this way and cost ~+0.0015 p_L. '
                  f'Tightening the integer widths would buy fractional precision here.', flush=True)

        for cls, (I, kneg, is_relu) in ActQuant._CLASSES.items():
            ActQuant._quantizers[cls] = (quantized_relu(ActQuant._bits, I) if is_relu
                                         else quantized_bits(ActQuant._bits, I, keep_negative=kneg))

        # Print the resulting fixed-point format for every class. Makes the actual arithmetic
        # visible in each run's log instead of implicit in a table two files away -- if a run
        # later looks wrong, the format it used is right there at the top of its own output.
        rows = []
        for cls, (I, kneg, is_relu) in sorted(ActQuant._CLASSES.items()):
            sign = 1 if (kneg and not is_relu) else 0
            frac = ActQuant._fractional_width(ActQuant._bits, I, kneg, is_relu)
            # ap_fixed<W, I> is sign-inclusive on the integer field, hence I + sign.
            rows.append(f'{cls}=ap_{"u" if not sign else ""}fixed<{ActQuant._bits},'
                        f'{I + sign}>(frac={frac})')
        print(f'[actquant] B={ActQuant._bits}  ' + '  '.join(rows), flush=True)

    @staticmethod
    def set_relu_integer(I):
        """Retune the decoder-ReLU integer width (default abs-max I=6; tighten to p99.9 I=4/5
        only once the model trains). Call BEFORE set_bits/build."""
        i, _, r = ActQuant._CLASSES['relu']
        ActQuant._CLASSES['relu'] = (I, False, True)

    @staticmethod
    def qa(x, cls):
        """Quantize an activation tensor at its point of use (no-op in FP32/disabled mode)."""
        if x is None or ActQuant._bits is None:
            return x
        return ActQuant._quantizers[cls](x)


# ---------------------------------------------------------------------------
# Patched methods. Each mirrors the original in CNNModel.py *exactly*, adding
# only `WeightQuant.q(...)` at the weight's point of use (before any matmul).
# ---------------------------------------------------------------------------

def _cnnkwe_get_mapped_weights(self, w, wmap):
    # CNNKernelWithEmbedding: quantizes kernel_weights_det_bits / det_evts
    # (evaluate() routes both weights through this method before its matmul).
    w = WeightQuant.q(w)
    if not self.is_symmetric or wmap is None:
        return w
    wgts_mapped = []
    for mm in wmap:
        jout = mm[0]
        ilist = mm[1]
        if ilist is None:
            wgts_mapped.append(w[:, jout])
        else:
            wgts_mapped.append(tf.gather(w[:, jout], ilist))
    return tf.stack(wgts_mapped, axis=1)


def _cnnkwe_get_mapped_bias(self, n):
    # CNNKernelWithEmbedding: quantizes kernel_bias (None when npol>1).
    kernel_bias = WeightQuant.q(self.kernel_bias)
    if kernel_bias is None or not self.is_symmetric or self.final_res_map is None:
        return tf.repeat(kernel_bias, n, axis=0) if kernel_bias is not None else None
    return tf.repeat(tf.gather(kernel_bias, self.final_res_map, axis=1), n, axis=0)


# CNNStateCorrelator: WEIGHTS are quantized (below). Its OUTPUT activation is deliberately
# NOT quantized here. At the r=3 default (use_exp_act=True) the correlator returns the x-like
# clip_exp result in [0, inf) -- the same ~10-decade x-domain tensor the Phase-3 lambda_min(C)
# result is about (non-positive on 5-19% of shots for the n>=3 instances, ranging up to ~4e4).
# x-like tensors are un-representable in fixed point, so this output is left FP32 and handled in
# Phase 4 (log-domain / LSE), exactly like the combiner's pre-log x-like intermediates. The
# alternate use_exp_act=False branch returns a p/f value res/(1+res) in (0,1), but that branch is
# not on the r=3 path and is not activation-quantized here; if it is ever enabled it would need
# ActQuant.qa(..., 'pf') on the correlator output.
def _cnnsc_get_mapped_weights(self, w, wmap):
    # CNNStateCorrelator: quantizes params_state_evolutions
    # (call() routes every use of that weight through this method before its matmul).
    w = WeightQuant.q(w)
    if wmap is None:
        return w
    wgts_mapped = []
    for mm in wmap:
        jout = mm[0]
        ilist = mm[1]
        if ilist is None:
            wgts_mapped.append(w[:, jout])
        else:
            wgts_mapped.append(tf.gather(w[:, jout], ilist))
    return tf.stack(wgts_mapped, axis=1)


def _cnnsc_get_mapped_bias(self, bias, n):
    # CNNStateCorrelator: quantizes params_b.
    bias = WeightQuant.q(bias)
    if bias is None or not self.is_symmetric or self.output_map is None:
        return tf.repeat(bias, n, axis=0) if bias is not None else None
    return tf.repeat(tf.gather(bias, self.output_map, axis=1), n, axis=0)


def _combiner_call(self, all_inputs):
    # RCNNKernelCombiner.call, mirrored; only change = WeightQuant.q on the three
    # translation weights (frac / phase / inverter) at their point of use. Shape
    # reads (frac_params.shape[0], phase_params.shape[0]) keep the raw params.
    kernel_outputs = self.kernel_collector(all_inputs)

    data_qubit_idxs_preds = []
    for udkc in self.unique_dqubit_kernel_contribs:
        data_qubit_idxs = udkc[1]
        frac_params = udkc[2]
        phase_params = udkc[3]
        inverter_params = udkc[4]
        frac_values = None
        two_phase_values = None
        inverter_values = None
        if frac_params is not None:
            frac_values = ActQuant.qa(
                self.frac_activation(VariableBounds.clip_zlike(WeightQuant.q(frac_params))), 'pf')
        if phase_params is not None:
            # the reference architecture Eq. 3.1: the phase term is 2*c_phi with c_phi = tanh(...) in (-1,1).
            # Quantize c_phi at its natural (-1,1) range ('cphi', integer=0), THEN multiply by 2 --
            # the *2 is a lossless power-of-2 shift, so it needs no quantizer and wastes no bit.
            two_phase_values = ActQuant.qa(
                self.phase_activation(WeightQuant.q(phase_params)), 'cphi') * 2
        if inverter_params is not None:
            inverter_values = self.inverter_activation(WeightQuant.q(inverter_params)) * 2  # dormant at r=3

        for idq_idkqs in data_qubit_idxs:
            idq = idq_idkqs[0]
            idkqs = idq_idkqs[1]
            sum_kouts = None
            sum_inputs = []
            for iktype, idkq in enumerate(idkqs):
                kout_list = []
                for ikq_idxkq in idkq:
                    ikq = ikq_idxkq[0]
                    idxkq = ikq_idxkq[1]
                    single_kernout = kernel_outputs[ikq][:, idxkq]
                    if inverter_values is not None:
                        single_kernout = tf.math.pow(single_kernout, inverter_values[iktype])
                    kout_list.append(single_kernout)
                kout = None
                for iik in range(len(kout_list)):
                    kki = kout_list[iik]
                    if kout is None:
                        kout = kki
                    else:
                        kout += kki
                    for jjk in range(iik + 1, len(kout_list)):
                        kkj = kout_list[jjk]
                        kout += tf.math.sqrt(kki * kkj) * 2
                kout /= len(kout_list)
                if frac_params is not None:
                    frac = None
                    for ifrac in range(min(frac_params.shape[0], iktype + 1)):
                        frac_tmp = frac_values[ifrac]
                        if ifrac != iktype:
                            frac_tmp = 1. - frac_tmp
                        if frac is None:
                            frac = frac_tmp
                        else:
                            frac = frac * frac_tmp
                    kout = kout * frac
                if sum_kouts is None:
                    sum_kouts = kout
                else:
                    sum_kouts = sum_kouts + kout
                sum_inputs.append(kout)
            n_sum_inputs = len(sum_inputs)
            if phase_params is not None:
                if n_sum_inputs * (n_sum_inputs - 1) // 2 != phase_params.shape[0]:
                    raise RuntimeError(f"Number of phase parameters {phase_params.shape[0]} does not match the number of inputs {n_sum_inputs}.")
                iphase = 0
                for idx_i1 in range(n_sum_inputs):
                    for idx_i2 in range(idx_i1 + 1, n_sum_inputs):
                        two_cos_phase = two_phase_values[iphase]
                        sum_kouts = sum_kouts + tf.sqrt(sum_inputs[idx_i1] * sum_inputs[idx_i2]) * two_cos_phase
                        iphase += 1
            # x-like intermediates (single_kernout, kout, sum_inputs, pre-log sum_kouts) stay FLOAT
            # (un-representable in fixed point -> Phase 4 LSE). Quantize only the z-like output z''.
            sum_kouts = ActQuant.qa(tf.math.log(VariableBounds.clip_exp(sum_kouts)), 'zlike')
            data_qubit_idxs_preds.append([idq, sum_kouts])
    data_qubit_idxs_preds.sort()

    data_qubit_final_preds = tf.concat(
        [tf.reshape(dqp[1], shape=(arrayops_shape(dqp[1], 0), -1)) for dqp in data_qubit_idxs_preds],
        axis=1
    )
    return self.eval_final_data_qubit_pred_layer(data_qubit_final_preds)


def _combiner_eval_final(self, data_qubit_final_preds):
    # RCNNKernelCombiner: quantizes NonUniformResponseAdj (None unless has_nonuniform_response).
    nonuniform_response_adj = WeightQuant.q(self.nonuniform_response_adj)
    if nonuniform_response_adj is not None:
        data_qubit_final_preds = data_qubit_final_preds + tf.repeat(
            nonuniform_response_adj, arrayops_shape(data_qubit_final_preds, 0), axis=0)
    return data_qubit_final_preds


def _statedecoder_init(self, code_distance, hidden_specs, do_all_data_qubits):
    # StateDecoder.__init__, mirrored; stock Dense -> WeightQuant.make_dense (QDense when quantizing).
    tf.keras.layers.Layer.__init__(self)
    self.code_distance = code_distance
    self.hidden_specs = hidden_specs
    self.do_all_data_qubits = do_all_data_qubits

    noutputs = 1 if not self.do_all_data_qubits else self.code_distance ** 2
    self.layers_decoder = []
    if hidden_specs is not None:
        for hl in hidden_specs:
            if type(hl) == int:
                self.layers_decoder.append(WeightQuant.make_dense(hl, activation="relu"))
            elif type(hl) == dict:
                is_activation = hl["is_activation"] if "is_activation" in hl else False
                if not is_activation:
                    n_nodes = hl["n_nodes"]
                    has_activation = hl["has_activation"] if "has_activation" in hl else True
                    activation = None
                    if has_activation:
                        activation = hl["activation"] if "activation" in hl else "relu"
                    self.layers_decoder.append(WeightQuant.make_dense(n_nodes, activation=activation))
                else:
                    self.layers_decoder.append(hl["activation"])
    if len(self.layers_decoder) > 0:
        self.layers_decoder.append(WeightQuant.make_dense(noutputs, activation="sigmoid"))
    else:
        self.layers_decoder.append(tf.keras.layers.Activation('sigmoid'))


def _statedecoder_call(self, inputs):
    # StateDecoder.call, mirrored; adds ActQuant on the z-like input (dec_in) and on each hidden
    # ReLU output (all layers but the final sigmoid, which is the decision -> left FP). No-op when
    # ActQuant disabled => byte-identical to the original (the identity check).
    x = ActQuant.qa(inputs, 'zlike')
    last = len(self.layers_decoder) - 1
    for i, layer in enumerate(self.layers_decoder):
        x = layer(x)
        if i < last:
            x = ActQuant.qa(x, 'relu')
    return x


def _make_output_quant_call(orig, cls):
    """Wrap a layer's call so its OUTPUT is activation-quantized (no-op when disabled).

    The wrapper advertises **kw, so Keras' Layer.__call__ passes `training` (and possibly `mask`)
    into it. The wrapped embedder call() methods take only their input tensor and do not accept
    those kwargs, so we drop them before forwarding. (The originals ignore training/mask anyway --
    they have no training-specific behaviour.)"""
    def call(self, *a, **kw):
        kw.pop('training', None)
        kw.pop('mask', None)
        return ActQuant.qa(orig(self, *a, **kw), cls)
    return call


# ---------------------------------------------------------------------------
# Installation (idempotent). Keeps a handle on the originals so quantization
# can be fully undone within a process if ever needed (restore_originals()).
# ---------------------------------------------------------------------------

_ORIGINALS = {}
_PATCHED = False


def _install_patches():
    global _PATCHED
    if _PATCHED:
        return
    _ORIGINALS.update({
        (CNNKernelWithEmbedding, 'get_mapped_weights'): CNNKernelWithEmbedding.get_mapped_weights,
        (CNNKernelWithEmbedding, 'get_mapped_bias'): CNNKernelWithEmbedding.get_mapped_bias,
        (CNNStateCorrelator, 'get_mapped_weights'): CNNStateCorrelator.get_mapped_weights,
        (CNNStateCorrelator, 'get_mapped_bias'): CNNStateCorrelator.get_mapped_bias,
        (RCNNKernelCombiner, 'call'): RCNNKernelCombiner.call,
        (RCNNKernelCombiner, 'eval_final_data_qubit_pred_layer'): RCNNKernelCombiner.eval_final_data_qubit_pred_layer,
        (StateDecoder, '__init__'): StateDecoder.__init__,
        (StateDecoder, 'call'): StateDecoder.call,
        (DetectorBitStateEmbedder, 'call'): DetectorBitStateEmbedder.call,
        (DetectorEventStateEmbedder, 'call'): DetectorEventStateEmbedder.call,
        (TripletStateProbEmbedder, 'call'): TripletStateProbEmbedder.call,
    })
    CNNKernelWithEmbedding.get_mapped_weights = _cnnkwe_get_mapped_weights
    CNNKernelWithEmbedding.get_mapped_bias = _cnnkwe_get_mapped_bias
    CNNStateCorrelator.get_mapped_weights = _cnnsc_get_mapped_weights
    CNNStateCorrelator.get_mapped_bias = _cnnsc_get_mapped_bias
    RCNNKernelCombiner.call = _combiner_call
    RCNNKernelCombiner.eval_final_data_qubit_pred_layer = _combiner_eval_final
    StateDecoder.__init__ = _statedecoder_init
    StateDecoder.call = _statedecoder_call
    # embedder OUTPUT activation-quant (Detector{Bit,Event} -> 'embed'; Triplet -> 'pf')
    DetectorBitStateEmbedder.call = _make_output_quant_call(_ORIGINALS[(DetectorBitStateEmbedder, 'call')], 'embed')
    DetectorEventStateEmbedder.call = _make_output_quant_call(_ORIGINALS[(DetectorEventStateEmbedder, 'call')], 'embed')
    TripletStateProbEmbedder.call = _make_output_quant_call(_ORIGINALS[(TripletStateProbEmbedder, 'call')], 'pf')
    _PATCHED = True


def restore_originals():
    """Undo the patches within this process (mainly for tests)."""
    global _PATCHED
    for (cls, name), fn in _ORIGINALS.items():
        setattr(cls, name, fn)
    _ORIGINALS.clear()
    _PATCHED = False
    WeightQuant.set_bits(None)
    ActQuant.set_bits(None)


def enable_weight_quantization(weight_bits):
    """Set the sweep's weight bit-width and install the quantized wrappers.
    weight_bits None or >=32 keeps everything full precision (patches stay inert)."""
    WeightQuant.set_bits(weight_bits)
    _install_patches()


def enable_activation_quantization(act_bits):
    """Set the activation word length B (Phase 2a) and install the wrappers.
    act_bits None or >=32 => activations stay FP32 (the anchor / identity check)."""
    ActQuant.set_bits(act_bits)
    _install_patches()


def build_quantized_rcnn(weight_bits, *args, act_bits=None, **kwargs):
    """Convenience: enable weight (and optionally activation) quantization, then build
    FullRCNNModel. Positional/keyword args are forwarded verbatim to FullRCNNModel.
    act_bits=None (default) => activations FP32, so existing weight-only callers are unchanged
    and the model reproduces the w6/act-FP32 anchor bit-for-bit (the Phase-2a identity check)."""
    enable_weight_quantization(weight_bits)
    enable_activation_quantization(act_bits)
    return FullRCNNModel(*args, **kwargs)
