import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.models import Model
from circuit_partition import *


class CNNEmbedder(Layer):
  """
  CNNEmbedder: Convert the input data into a quadratic form with trainable probability assignments.
  The input needs to have dimensions (batch size, rounds>2, number of ancillas).
  The output will have dimensions (batch size, rounds-2, ndims),
  where ndims = n_ancillas if npol=1, and ndims = n_ancillas*(n_ancillas+1)/2 otherwise.
  - If npol>1, there are only 3 possible values for the state of diagonal terms: -1, 0, 1.
  For that reason, there is a simple map of -1 -> 0, 0 -> p, 1 -> 1, where p is the only probability parameter for training.
  For non-diagonal terms, there are 6 possible values: -1x-1, -1x0, -1x1, 0x0, 0x1, and 1x1.
  Their probability parameter map is -1x-1 -> 0, -1x0 -> p1*p2*p3*p4, -1x1 -> p1*p2*p3, 0x0 -> p1*p2, 0x1 -> p1, and 1x1 -> 1.
  """
  def __init__(
      self,
      distance,
      rounds,
      npol,
      append_name="",
      **kwargs
    ):
    super(CNNEmbedder, self).__init__(**kwargs)
    self.distance = distance
    self.rounds = rounds
    self.npol = npol
    self.n_ancillas = (self.distance**2 - 1)
    self.embed_label = f"CNNEmbedder{append_name}_d{self.distance}_r{self.rounds}_npol{self.npol}"

    self.state_tracker = None
    self.delta_tracker = None

    ndim1 = self.n_ancillas # Number of ancilla measurements
    self.triangular_polmap = None
    self.state_embedding_diag = None
    self.state_embedding_nondiag = None
    if self.npol>1:
      nondiag = []
      diag = []
      for iy in range(ndim1):
        for ix in range(iy, ndim1):
          if ix!=iy:
            nondiag.append(ix+iy*ndim1)
          else:
            diag.append(ix+iy*ndim1)
      self.triangular_polmap = diag
      self.triangular_polmap.extend(nondiag)
      ndim1 = ndim1*(ndim1+1)//2      

      self.embedding_params_diag = self.add_weight(
        name=f"{self.embed_label}_embedded_params_diag",
        shape=[ 1, self.n_ancillas ], # Only state = 0 needs a probability assignment; state = -1 -> 0, and state = 1 -> 1
        initializer='zeros',
        trainable=True
      )
      self.embedding_params_nondiag = self.add_weight(
        name=f"{self.embed_label}_embedded_params_nondiag",
        shape=[ 1, ndim1-self.n_ancillas, 4 ], # States -1x0, -1x1, 0x0, 0x1 need probability assignment; state = -1x-1 -> 0, and state = 1x1 -> 1
        initializer='zeros',
        trainable=True
      )


  def embed_pol_state(self, res_diag, res_nondiag, embedding_diag_tr, embedding_nondiag_tr):
    # Diagonal terms
    res_diag = tf.cast(res_diag, tf.float32)
    res_diag = res_diag[:,:,0] + res_diag[:,:,1]*embedding_diag_tr
    # Non-diagonal terms
    res_nondiag = tf.cast(res_nondiag, tf.float32)
    res_nondiag = res_nondiag[:,:,0] + tf.reduce_sum(res_nondiag[:,:,1:5]*embedding_nondiag_tr, axis=2)
    res = tf.concat([res_diag, res_nondiag], axis=1)
    return res


  def transform_raw_state(self, x, embedding_diag_tr, embedding_nondiag_tr):
    res = None
    if self.npol==1:
      res = tf.cast(x, tf.float32)
    else:
      xx = x + 3 # (-1, 0, 1) -> (2, 3, 4)
      res = tf.matmul(
          tf.cast(tf.reshape(xx, shape=(x.shape[0], x.shape[1], 1)), tf.int16),
          tf.cast(tf.reshape(xx, shape=(x.shape[0], 1, x.shape[1])), tf.int16)
      )
      res = tf.gather(tf.reshape(res, shape=(x.shape[0], -1)), self.triangular_polmap, axis=1)
      res_diag = res[:,0:self.n_ancillas]
      res_nondiag = res[:,self.n_ancillas:]
      # res has values (4, 8, 16 | 6, 9, 12) now
      is16_diag = tf.equal(res_diag, 16)
      is9_diag = tf.equal(res_diag, 9)
      is16_nondiag = tf.equal(res_nondiag, 16)
      is12_nondiag = tf.equal(res_nondiag, 12)
      is9_nondiag = tf.equal(res_nondiag, 9)
      is8_nondiag = tf.equal(res_nondiag, 8)
      is6_nondiag = tf.equal(res_nondiag, 6)
      res_diag = tf.stack([is16_diag, is9_diag], axis=2)
      res_nondiag = tf.stack([is16_nondiag, is12_nondiag, is9_nondiag, is8_nondiag, is6_nondiag], axis=2)
      res = self.embed_pol_state(res_diag, res_nondiag, embedding_diag_tr, embedding_nondiag_tr)
    return res


  def get_transformed_embedding_params(self, n):
    if self.npol>1:
      embedding_diag_tr = tf.repeat(tf.math.sigmoid(self.embedding_params_diag), n, axis=0)
      f12 = tf.math.sigmoid(self.embedding_params_nondiag[:,:,0])
      f9 = tf.math.sigmoid(self.embedding_params_nondiag[:,:,1])*f12
      f8 = tf.math.sigmoid(self.embedding_params_nondiag[:,:,2])*f9
      f6 = tf.math.sigmoid(self.embedding_params_nondiag[:,:,3])*f8
      embedding_nondiag_tr = tf.repeat(tf.stack([f12, f9, f8, f6], axis=2), n, axis=0)
      return embedding_diag_tr, embedding_nondiag_tr
    return None, None


  def get_raw_state(self, input):
    data_err = input[:,0,:] + input[:,2,:] - input[:,0,:]*input[:,2,:]*2 # 001/011/110/100-like sequences
    measure_err = input[:,1,:]*(1 - (input[:,0,:] + input[:,2,:])) + input[:,0,:]*input[:,2,:] # 010/101-like sequences

    self.delta_tracker = self.delta_tracker*(1-measure_err*2)
    self.state_tracker = tf.math.minimum(
      tf.maximum(
        self.state_tracker + self.delta_tracker*data_err,
        -tf.ones_like(self.state_tracker)
      ),
      tf.ones_like(self.state_tracker)
    )
    self.delta_tracker = self.delta_tracker*(1-self.state_tracker*self.state_tracker*(1-measure_err)) - self.state_tracker*(1-measure_err)

    return self.state_tracker
  

  def get_transformed_state(self, input, embedding_diag_tr, embedding_nondiag_tr):
    res = []
    for rr in range(self.rounds-2):
      x = input[:,rr:rr+3,:]
      x = self.get_raw_state(x)
      x = self.transform_raw_state(x, embedding_diag_tr, embedding_nondiag_tr)
      res.append(x)
    res = tf.stack(res, axis=1) # (n, r-2, n_ancillas) dimensions
    return res


  def reset_tracker_states(self, n):
    self.state_tracker = tf.constant([[ -1 for _ in range(self.n_ancillas) ] for _ in range(n)], dtype=tf.int8)
    self.delta_tracker = tf.constant([[ 1 for _ in range(self.n_ancillas) ] for _ in range(n)], dtype=tf.int8)


  def call(self, input):
    n = input.shape[0]
    self.reset_tracker_states(n)
    embedding_diag_tr, embedding_nondiag_tr = self.get_transformed_embedding_params(n)
    return self.get_transformed_state(input, embedding_diag_tr, embedding_nondiag_tr)


class CNNKernel(Layer):
  def __init__(
      self,
      kernel_type, kernel_distance, rounds,
      npol=1,
      do_all_data_qubits = False,
      include_det_evts = True,
      n_remove_last_det_evts = 0,
      **kwargs
    ):
    super(CNNKernel, self).__init__(**kwargs)
    self.kernel_distance = kernel_distance
    self.rounds = rounds
    self.npol = npol
    self.do_all_data_qubits = do_all_data_qubits
    self.include_det_evts = include_det_evts
    self.n_remove_last_det_evts = n_remove_last_det_evts
    self.n_ancillas = (self.kernel_distance**2 - 1)

    constraint_label = f"{kernel_type[0]}_{kernel_type[1]}"
    num_outputs = 1
    if self.do_all_data_qubits:
      if kernel_type[0]==0 and kernel_type[1]==0:
        num_outputs = (self.kernel_distance**2 + 1)//2
      else:
        num_outputs = self.kernel_distance**2

    ndim1 = self.n_ancillas*rounds # Number of ancilla measurements
    ndim2 = 0 # Number of detector events
    if self.include_det_evts:
      ndim2 += self.n_ancillas//2 + self.n_ancillas*(rounds-1) # Number of detector event bits within each round
      ndim2 += self.n_ancillas//2 - self.n_remove_last_det_evts
    self.triangular_polmap_det_bits = None
    self.triangular_polmap_det_evts = None
    if self.npol>1:
      self.triangular_polmap_det_bits = []
      for iy in range(ndim1):
        for ix in range(iy, ndim1):
          self.triangular_polmap_det_bits.append(ix+iy*ndim1)
      self.triangular_polmap_det_evts = []
      for iy in range(ndim2):
        for ix in range(iy, ndim2):
          self.triangular_polmap_det_evts.append(ix+iy*ndim2)
      ndim1 = ndim1*(ndim1+1)//2
      ndim2 = ndim2*(ndim2+1)//2

    self.ndims = [ [ ndim1, num_outputs ], [ ndim2, num_outputs ] ]
    self.kernel_weights_det_bits = self.add_weight(
      name=f"CNNkernel{self.kernel_distance}_{constraint_label}_w_det_bits",
      shape=self.ndims[0],
      initializer='zeros',
      trainable=True
    )
    self.kernel_weights_det_evts = None
    if self.include_det_evts:
      self.kernel_weights_det_evts = self.add_weight(
        name=f"CNNkernel{self.kernel_distance}_{constraint_label}_w_det_evts",
        shape=self.ndims[1],
        initializer='zeros',
        trainable=True
      )
    self.kernel_bias = None
    self.kernel_activation = None
    if self.npol<=1:
      self.kernel_bias = self.add_weight(
        name=f"CNNkernel{self.kernel_distance}_{constraint_label}_b",
        shape=[ num_outputs ],
        initializer='zeros',
        trainable=True
      )
    self.kernel_activation = tf.keras.activations.sigmoid

  def build(self, input_shape):
    pass

  def transform_inputs(self, x, tmap):
    res = None
    if self.npol==1:
      res = x
    else:
      res = tf.matmul(
          tf.cast(tf.reshape((x+1), shape=(x.shape[0], x.shape[1], 1)), tf.int16),
          tf.cast(tf.reshape((x+1), shape=(x.shape[0], 1, x.shape[1])), tf.int16)
      )
      res = tf.gather(tf.reshape(res, (x.shape[0], -1)), tmap, axis=1)
      res = (-res*res+res*9-14)/6 # (1, 2, 4) -> (-1, 0, 1)
    return tf.cast(res, tf.float32)

  def evaluate(self, bits, do_evts):
    w = None
    tmap = None
    if not do_evts:
      w = self.kernel_weights_det_bits
      tmap = self.triangular_polmap_det_bits
    else:
      w = self.kernel_weights_det_evts
      tmap = self.triangular_polmap_det_evts
    return tf.matmul(self.transform_inputs(bits, tmap), w)

  def call(self, inputs):
    res = self.evaluate(inputs[0], False)
    if self.include_det_evts:
      res = res + self.evaluate(inputs[1], True)
    if self.kernel_bias is not None:
      res = res + self.kernel_bias
    if self.kernel_activation is not None:
      res = self.kernel_activation(res)
    return res
  

class FullCNNModel(Model):
  def __init__(
      self,
      obs_type, code_distance, kernel_distance, rounds,
      hidden_specs,
      npol = 1,
      do_all_data_qubits = False,
      extended_kernel_output = True,
      include_det_evts = True,
      include_last_kernel_dets = False,
      include_last_dets = True,
      has_nonuniform_response = False,
      KernelProcessor = CNNKernel,
      **kwargs
    ):
    super(FullCNNModel, self).__init__(**kwargs)
    self.obs_type = obs_type
    self.code_distance = code_distance
    self.kernel_distance = kernel_distance
    self.kernel_half_distance = kernel_distance//2
    self.n_kernel_last_det_evts = (self.kernel_distance**2-1)//2
    self.nshifts = self.code_distance - self.kernel_distance + 1
    self.rounds = rounds
    self.npol = npol
    self.do_all_data_qubits = do_all_data_qubits
    self.extended_kernel_output = extended_kernel_output
    self.include_det_evts = include_det_evts
    self.include_last_kernel_dets = include_last_kernel_dets
    self.include_last_dets = include_last_dets
    self.has_nonuniform_response = has_nonuniform_response

    self.cnn_kernels = []
    self.unique_kernel_types = get_unique_kernel_types(self.kernel_distance, code_distance)
    for kernel_type in self.unique_kernel_types:
      n_remove_last_dets = 0
      kernel_parity = kernel_type[0]
      if self.include_det_evts:
        if self.include_last_kernel_dets:
          if self.obs_type=="ZL":
            if kernel_parity[0]==0:
              n_remove_last_dets = 2
            elif kernel_parity[0]==1 and self.code_distance>self.kernel_distance:
              n_remove_last_dets = 1
            elif kernel_parity[0]==-1:
              n_remove_last_dets = 1
          elif self.obs_type=="XL":
            if kernel_parity[1]==0:
              n_remove_last_dets = 2
            elif kernel_parity[1]==1 and self.code_distance>self.kernel_distance:
              n_remove_last_dets = 1
            elif kernel_parity[1]==-1:
              n_remove_last_dets = 1
        else:
          n_remove_last_dets = (self.kernel_distance**2-1)//2

      self.cnn_kernels.append(
        KernelProcessor(
          kernel_parity,
          self.kernel_distance,
          self.rounds,
          self.npol,
          self.do_all_data_qubits or self.extended_kernel_output,
          self.include_det_evts,
          n_remove_last_dets
        )
      )
    
    self.translation_coef_transform = Dense(self.nshifts)
    self.translation_coef_transform_act = tf.keras.layers.Activation('sigmoid')

    dqubit_kernel_contribs = [ [] for _ in range(self.code_distance**2) ]
    for shifty in range(self.nshifts):
      for shiftx in range(self.nshifts):
        ikernel = shiftx+shifty*self.nshifts
        ktype = None
        is_symmetric = False
        for iktype, kernel_type in enumerate(self.unique_kernel_types):
          if ikernel in kernel_type[1]:
            ktype = iktype
            if kernel_type[0][0]==0 and kernel_type[0][1]==0:
              is_symmetric = True
            break
        _, _, flip_x, flip_y = get_kernel_parity_flips(self.nshifts, shiftx, shifty)
        for ky in range(-self.kernel_half_distance,self.kernel_half_distance+1):
          iy = ky if not flip_y else -ky
          jy = self.kernel_half_distance+iy
          if shifty+jy<0:
            continue
          for kx in range(-self.kernel_half_distance,self.kernel_half_distance+1):
            ix = kx if not flip_x else -kx
            jx = self.kernel_half_distance+ix
            if shiftx+jx<0:
              continue
            ox = kx
            oy = ky
            if is_symmetric and (ox<oy or (ox==oy and ox>0)):
              ox = -ox
              oy = -oy
            idx_kqubit = (oy+self.kernel_half_distance)*self.kernel_distance + (ox+self.kernel_half_distance)
            if is_symmetric:
              idx_kqubit = idx_kqubit - (oy+self.kernel_half_distance)*(oy+self.kernel_half_distance+1)//2
            idx_dqubit = (shiftx+jx) + (shifty+jy)*self.code_distance
            found = False
            for dqkcs in dqubit_kernel_contribs[idx_dqubit]:
              if dqkcs[0][0]==ktype and dqkcs[0][1]==idx_kqubit:
                dqkcs[1].append(ikernel)
                found = True
                break
            if not found:
              dqubit_kernel_contribs[idx_dqubit].append([[ ktype, idx_kqubit ], [ ikernel ]]) # Kernel type index, qubit index within kernel
    for dqkcs in dqubit_kernel_contribs:
      dqkcs.sort(key=lambda x: x[0][0]*self.kernel_distance**2 + x[0][1])
    self.unique_dqubit_kernel_contribs = []
    for idq, dqkcs in enumerate(dqubit_kernel_contribs):
      type_contribs = []
      kernel_idxs = []
      for ddd in dqkcs:
        type_contribs.append(ddd[0])
        kernel_idxs.append(ddd[1])
      found = False
      #print(f"Data qubit {idq} -> ktype={type_contribs}, kernel indices = {kernel_idxs}")
      dq_kernelidx_map = [idq, kernel_idxs]
      for udkc in self.unique_dqubit_kernel_contribs:
        if type_contribs==udkc[0]:
          found = True
          udkc[1].append(dq_kernelidx_map)
          break
      if not found:
        self.unique_dqubit_kernel_contribs.append([type_contribs, [ dq_kernelidx_map ]])

    print(f"Number of unique contributions: {len(self.unique_dqubit_kernel_contribs)}")

    total_nfracs = 0
    total_nphases = 0
    self.frac_params = []
    self.phase_params = []
    for iudkc, udkc in enumerate(self.unique_dqubit_kernel_contribs):
      #print(f"Kernel type = {udkc[0]}")
      #for uuu in udkc[1]:
      #  print(f"- Data qubit {uuu[0]} maps to kernels {uuu[1]}")
      np = len(udkc[0])
      nfr = np-1
      nph = np*(np-1)//2
      total_nfracs += nfr
      total_nphases += nph
      if nfr>0:
        udkc.append(
          self.add_weight(
            name=f"TranslationFrac_{iudkc}",
            shape=[ nfr ],
            initializer='zeros',
            trainable=True
          )
        )
      else:
        udkc.append(None)
      if nph>0:
        udkc.append(
          self.add_weight(
            name=f"TranslationPhase_{iudkc}",
            shape=[ nph ],
            initializer='zeros',
            trainable=True
          )
        )
      else:
        udkc.append(None)
    print(f"Total number of fractions: {total_nfracs}")
    print(f"Total number of phases: {total_nphases}")
    self.frac_activation = tf.keras.activations.sigmoid # We actually need cos(phi), so there is an activation function
    self.phase_activation = tf.keras.activations.tanh # We actually need cos(phi), so there is an activation function

    self.nonuniform_response_adj = None
    if self.has_nonuniform_response:
      self.nonuniform_response_adj = self.add_weight(
        name=f"NonUniformResponseAdj",
        shape=[ 1, self.code_distance**2 ],
        initializer='zeros',
        trainable=True
      )

    self.noutputs_final = 1 if not self.do_all_data_qubits else self.code_distance**2
    self.first_dense_nout = None
    self.upper_layers = []
    for hs in hidden_specs:
      if self.first_dense_nout is None:
        self.first_dense_nout = hs
      self.upper_layers.append(Dense(hs))
      self.upper_layers.append(tf.keras.layers.Activation('relu'))

    if self.first_dense_nout is None: # only happens if there are no hidden layers
      self.first_dense_nout = self.noutputs_final
    self.data_qubit_pred_eval_layer = Dense(self.first_dense_nout, use_bias=False)
    self.upper_layers.append(Dense(self.noutputs_final))
    self.upper_layers.append(tf.keras.layers.Activation('sigmoid'))

  def eval_final_data_qubit_pred_layer(self, data_qubit_final_preds):
    # We assume data_qubit_final_preds is flat along axis=1
    if self.nonuniform_response_adj is not None:
      data_qubit_final_preds = data_qubit_final_preds + tf.repeat(self.nonuniform_response_adj, data_qubit_final_preds.shape[0], axis=0)
    return self.data_qubit_pred_eval_layer(data_qubit_final_preds)


  def call(self, all_inputs):
    det_bits = all_inputs[0]
    det_evts = all_inputs[1]
    translation_coefs = all_inputs[2]
    final_det_evts = all_inputs[3]

    kernel_outputs = dict()
    predictor_inputs = []

    predictor_inputs.append(tf.cast(translation_coefs, tf.float32))
    if self.include_det_evts and self.include_last_dets:
      predictor_inputs.append(tf.cast(final_det_evts, tf.float32))
    predictor_inputs = tf.concat(predictor_inputs, axis=1)

    translation_coefs_transformed = self.translation_coef_transform_act(self.translation_coef_transform(translation_coefs))
    translation_coefs_transformed = tf.reshape(translation_coefs_transformed, (translation_coefs_transformed.shape[0], translation_coefs_transformed.shape[1], 1))
    if self.extended_kernel_output:
      translation_coefs_transformed = tf.repeat(translation_coefs_transformed, self.kernel_distance**2, axis=2)

    for i, cnn_kernel in enumerate(self.cnn_kernels):
      kernel_parity = self.unique_kernel_types[i][0]
      kernel_idxs = self.unique_kernel_types[i][1]
      for k in kernel_idxs:
        kernel_input = None
        det_bits_kernel = det_bits[:,k]
        if self.include_det_evts:
          det_evts_kernel = det_evts[:,k,0:-self.n_kernel_last_det_evts]
          if self.include_last_kernel_dets:
            det_evts_kernel_end = det_evts[:,k,-self.n_kernel_last_det_evts:]
            if self.obs_type=="ZL":
              if kernel_parity[0]==0:
                det_evts_kernel_end = det_evts_kernel_end[:,1:-1]
              elif kernel_parity[0]==1 and self.code_distance>self.kernel_distance:
                det_evts_kernel_end = det_evts_kernel_end[:,:-1]
              elif kernel_parity[0]==-1:
                det_evts_kernel_end = det_evts_kernel_end[:,1:]
            elif self.obs_type=="XL":
              if kernel_parity[1]==0:
                det_evts_kernel_end = det_evts_kernel_end[:,1:-1]
              elif kernel_parity[1]==1 and self.code_distance>self.kernel_distance:
                det_evts_kernel_end = det_evts_kernel_end[:,:-1]
              elif kernel_parity[1]==-1:
                det_evts_kernel_end = det_evts_kernel_end[:,1:]
            det_evts_kernel = tf.concat([det_evts_kernel, det_evts_kernel_end], axis=1)
          kernel_input = [ det_bits_kernel, det_evts_kernel ]
        else:
          kernel_input = [ det_bits_kernel ]
        kernel_output = cnn_kernel(kernel_input)
        if self.extended_kernel_output:
          shift_x = k % self.nshifts
          shift_y = k // self.nshifts
          k_shift = shift_y if self.obs_type=="ZL" else shift_x
          kernel_output = tf.math.pow(
              kernel_output,
              1.-translation_coefs_transformed[:,k_shift,0:kernel_output.shape[1]]
            )*tf.math.pow(
              1.-kernel_output,
              translation_coefs_transformed[:,k_shift,0:kernel_output.shape[1]]
            )
        kernel_output = tf.clip_by_value(kernel_output, 1e-6, 1.-1e-6)
        kernel_outputs[k] = [i, kernel_output/(1.-kernel_output)] # [Kernel unique type index, transformed kernel output]
    
    data_qubit_idxs_preds = []
    for udkc in self.unique_dqubit_kernel_contribs:
      kernel_type_contribs = udkc[0]
      data_qubit_idxs = udkc[1]
      frac_params = udkc[2]
      phase_params = udkc[3]
      frac_values = None
      phase_values = None
      if frac_params is not None:
        frac_values = tf.clip_by_value(self.frac_activation(frac_params), 1e-6, 1.-1e-6)
      if phase_params is not None:
        phase_values = self.phase_activation(phase_params)

      for idq_idkqs in data_qubit_idxs:
        idq = idq_idkqs[0]
        idkqs = idq_idkqs[1]
        sum_kouts = None
        sum_inputs = []
        for iktype, idkq in enumerate(idkqs):
          ktype = kernel_type_contribs[iktype]
          kout = None
          for ikq in idkq:
            if kout is None:
              kout = kernel_outputs[ikq][1][:,ktype[1]]
            else:
              kout = kout + kernel_outputs[ikq][1][:,ktype[1]]
          if frac_params is not None:
            frac = None
            for ifrac in range(min(frac_params.shape[0],iktype+1)):
              frac_tmp = frac_values[ifrac]
              if ifrac!=iktype:
                frac_tmp = 1.-frac_tmp
              if frac is None:
                frac = frac_tmp
              else:
                frac = frac*frac_tmp
            kout = kout*frac
          if sum_kouts is None:
            sum_kouts = kout
          else:
            sum_kouts = sum_kouts + kout
          sum_inputs.append(kout)
        n_sum_inputs = len(sum_inputs)
        if phase_params is not None:
          if n_sum_inputs*(n_sum_inputs-1)//2!=phase_params.shape[0]:
            raise RuntimeError(f"Number of phase parameters {phase_params.shape[0]} does not match the number of inputs {n_sum_inputs}.")
          iphase = 0
          for idx_i1 in range(n_sum_inputs):
            for idx_i2 in range(idx_i1+1, n_sum_inputs):
              cos_phase = phase_values[iphase]
              iphase += 1
              sum_kouts = sum_kouts + 2*tf.sqrt(sum_inputs[idx_i1]*sum_inputs[idx_i2])*cos_phase
        #sum_kouts = sum_kouts/(1.+sum_kouts)
        sum_kouts = tf.math.log(tf.clip_by_value(sum_kouts, 1e-9, 1e9))
        data_qubit_idxs_preds.append([idq, sum_kouts])
    data_qubit_idxs_preds.sort()

    data_qubit_final_preds = tf.concat(
      [ tf.reshape(dqp[1], shape=(dqp[1].shape[0],-1)) for dqp in data_qubit_idxs_preds ],
      axis=1
    )
    eval_dqubit_preds_layer = None

    x = predictor_inputs
    for ll in self.upper_layers:
      x = ll(x)
      if eval_dqubit_preds_layer is None:
        eval_dqubit_preds_layer = self.eval_final_data_qubit_pred_layer(data_qubit_final_preds)
        x = x + eval_dqubit_preds_layer
    return x
