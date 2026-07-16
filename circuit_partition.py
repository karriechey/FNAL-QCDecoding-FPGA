import numpy as np
from copy import deepcopy
from utilities_arrayops import *
from types_cfg import get_types


def coord_to_idx(coords, d):
  _, _, idx_t, _ = get_types(d, 1, 1)
  return idx_t(coords[0] + (coords[1] - coords[0]%2)*(d+0.5))


def get_measure_qubits_ord(d):
  res = []
  for x in range(d+1):
    for y in range(d+1):
      on_boundary_1 = x == 0 or x == d
      on_boundary_2 = y == 0 or y == d
      parity = x % 2 != y % 2
      if (parity and on_boundary_1) or (not parity and on_boundary_2): continue
      coords = [2*x, 2*y]
      lbl = 'M:Z' if not parity else 'M:X'
      res.append([coord_to_idx(coords, d), lbl, coords])
  return res


def get_data_qubits_ord(d):
  res = []
  for x in range(d):
    for y in range(d):
      coords = [2*x+1, 2*y+1]
      is_ZL = y==0
      is_XL = x==0
      lbl = 'D'
      if is_ZL:
        lbl = lbl + ":ZL"
      if is_XL:
        lbl = lbl + ":XL"
      res.append([coord_to_idx(coords, d), lbl, coords])
  return res


def get_data_qubits_next_to_measure_qubits(d):
  data_qubits = sorted(get_data_qubits_ord(d))
  measure_qubits = sorted(get_measure_qubits_ord(d))
  res = []
  for mq in measure_qubits:
    coord_mq = mq[2]
    idxs_dq = []
    for idx_dq, dq in enumerate(data_qubits):
      coord_dq = dq[2]
      if abs(coord_dq[0]-coord_mq[0])*abs(coord_dq[1]-coord_mq[1]) == 1:
        idxs_dq.append(idx_dq)
    res.append(idxs_dq)
  return res


def get_kernel_parity_flips(n_shifts, shift_x, shift_y):
  """
  get_kernel_parity_flips: Find out which edges are at the boundary,
                    flip the kernel along x or y if necessary,
                    and extract the standard boundary configuration of the resulting kernel.
  Arguments:
  - n_shifts: The number of shifts in the kernel
  - shift_x: The shift along the x-axis
  - shift_y: The shift along the y-axis
  Returns:
  - Whether the flipped configuration has a boundary on the left (x=+1) or right (x=-1), or top (y=+1) or bottom (y=-1)
  - Whether the kernel is flipped along x or y
  """
  flip_x = ((shift_x + shift_y)%2)==1
  flip_y = False
  parity_x = (1 if shift_x==0 else (-1 if shift_x==n_shifts-1 else 0))
  if flip_x:
    parity_x = -parity_x
  parity_y = (1 if shift_y==0 else (-1 if shift_y==n_shifts-1 else 0))
  if (parity_x==0 and parity_y==-1) \
    or (parity_x==1 and parity_y==-1) \
    or (parity_x==-1 and parity_y!=1):
      flip_x = not flip_x
      flip_y = not flip_y
      parity_x = -parity_x
      parity_y = -parity_y
  return parity_x, parity_y, flip_x, flip_y


def get_unique_kernel_types(k, d):
  res = []
  n_shifts = d-k+1
  for shift_y in range(n_shifts):
    for shift_x in range(n_shifts):
      parity_x, parity_y, _, _ = get_kernel_parity_flips(n_shifts, shift_x, shift_y)
      has_parity = False
      for pp in res:
        if pp[0][0]==parity_x and pp[0][1]==parity_y:
          has_parity = True
          pp[1].append(shift_x + shift_y*n_shifts)
          break
      if not has_parity:
        res.append([[parity_x, parity_y], [shift_x + shift_y*n_shifts]])
  return res


def shift_frame(data_qubits_kxk, measure_qubits_kxk, k, d, shift_x, shift_y):
  _, _, flip_x, flip_y = get_kernel_parity_flips(d-k+1, shift_x, shift_y)
  res_data_qubits_kxk = None
  res_measure_qubits_kxk = None
  if data_qubits_kxk is not None:
    res_data_qubits_kxk = deepcopy(data_qubits_kxk)
    for q in res_data_qubits_kxk:
      q[2][0] = (q[2][0] if not flip_x else 2*k - q[2][0]) + 2*shift_x
      q[2][1] = (q[2][1] if not flip_y else 2*k - q[2][1]) + 2*shift_y
  if measure_qubits_kxk is not None:
    res_measure_qubits_kxk = deepcopy(measure_qubits_kxk)
    for q in res_measure_qubits_kxk:
      q[2][0] = (q[2][0] if not flip_x else 2*k - q[2][0]) + 2*shift_x
      q[2][1] = (q[2][1] if not flip_y else 2*k - q[2][1]) + 2*shift_y
  return res_data_qubits_kxk, res_measure_qubits_kxk


# Global dictionary to avoid reassembling the map of kernel groupings
dict_group_det_bits_kxk_ = {}

def group_det_bits_kxk(det_bits_dxd, d, r, k, use_rotated_z, data_bits_dxd=None, binary_t=np.int8, idx_t=np.int32, make_translation_map=True):
  """
  group_det_bits_kxk: Group the (d^2-1) detector bits into groups of size (k^2-1) for a kxk subset of the dxd surface code.
  Arguments:
  - det_bits_dxd: The measure bits for the dxd surface code as ordered by stim
  - d: The distance of the surface code
  - r: The number of rounds
  - k: The size of the subset of the surface code
  - use_rotated_z: Whether to use the ZL or XL boundaries in the rotated surface code
  - data_bits_dxd: The data bits for the dxd surface code as ordered by stim (optional)
  - binary_t: The data type for the bits
  - idx_t: The data type for the indices
  Returns:
  In order of appearance:
  - A 3D array of shape=[(d-k+1)^2, number of samples, r*(k^2-1)] for the detector bits
    that would be used in each kernel with each 2D [,:,:] subset ordered in a way consistent with stim
  - A 3D array of shape=[(d-k+1)^2, number of samples, k^2] for the data bits
  - A 3D array of shape=[(d-k+1)^2, number of samples, k] for the logical observables of each kernel
  - A 3D array of shape=[(d-k+1), number of samples, r] for the partial translation map
    of the kernel flip predictions to the original logical observable in the dxd surface code.
    In this translation map, the first dimension corresponds to the row (column) of an equivalent ZL (XL) observable.
  """
  opts = (d, r, k, use_rotated_z)
  cached_grouping_map = dict_group_det_bits_kxk_.get(opts, None)
  cached_det_bits_map = None if cached_grouping_map is None else cached_grouping_map[0]
  make_cached_det_bits_map = cached_det_bits_map is None
  if make_cached_det_bits_map:
    cached_det_bits_map = []
  cached_data_bits_map = None if cached_grouping_map is None else cached_grouping_map[1]
  cached_obs_bits_map = None if cached_grouping_map is None else cached_grouping_map[2]
  make_cached_data_bits_map = cached_data_bits_map is None and data_bits_dxd is not None
  if make_cached_data_bits_map:
    cached_data_bits_map = []
    cached_obs_bits_map = []
  cached_translation_idx_map = None if cached_grouping_map is None else cached_grouping_map[3]
  make_cached_translation_idx_map = cached_translation_idx_map is None and make_translation_map
  if make_cached_translation_idx_map:
    cached_translation_idx_map = []

  # Number of kernels along each dimension
  n_shifts = d-k+1

  det_bits_kxk_all = []
  data_bits_kxk_all = []
  obs_bits_kxk_all = []
  kernel_result_translation_map = None
  if make_translation_map:
    kernel_result_translation_map = arrayops_zeros(
      (n_shifts, arrayops_shape(det_bits_dxd, 0), r),
      dtype = convert_to_npdtype(binary_t) if type(data_bits_dxd)==np.ndarray else convert_to_tfdtype(binary_t)
    )
  
  # Measure qubits
  measure_qubits_ord_dxd = None
  measure_qubits_ord_kxk = None
  n_measure_dxd = None
  n_measure_kxk = None
  # Data qubits
  data_qubits_ord_dxd = None
  data_qubits_ord_kxk = None
  n_data_dxd = None
  n_data_kxk = None
  # Ordering of qubits by measurement sequence:
  # Measurement sequence is the same as the ordered sequence of qubit indices in stim
  # Measure qubits are measured r times, and then data qubits are measured once.
  measurement_sequence_data_dxd = None
  measurement_sequence_measure_dxd = None
  if make_cached_det_bits_map or make_cached_data_bits_map or make_cached_translation_idx_map:
    measure_qubits_ord_dxd = get_measure_qubits_ord(d)
    measure_qubits_ord_kxk = get_measure_qubits_ord(k)
    n_measure_dxd = len(measure_qubits_ord_dxd)
    n_measure_kxk = len(measure_qubits_ord_kxk)

    data_qubits_ord_dxd = get_data_qubits_ord(d)
    data_qubits_ord_kxk = get_data_qubits_ord(k)
    n_data_dxd = len(data_qubits_ord_dxd)
    n_data_kxk = len(data_qubits_ord_kxk)

    measurement_sequence_data_dxd = sorted(data_qubits_ord_dxd)
    measurement_sequence_measure_dxd = sorted(measure_qubits_ord_dxd)

  for shift_y in range(n_shifts):
    for shift_x in range(n_shifts):
      # Get the filtering indices to translate kxk kernel results to the dxd ZL or XL boundaries
      if make_translation_map and ((shift_y==0 and not use_rotated_z) or (shift_x==0 and use_rotated_z)):
        idx_kernel_row = (shift_y if use_rotated_z else shift_x)
        measure_boundary_filter = None
        if make_cached_translation_idx_map:
          parity_x, parity_y, flip_x, flip_y = get_kernel_parity_flips(n_shifts, shift_x, shift_y)
          is_top = (parity_y!=-1 and not flip_y)
          is_left = (parity_x!=-1 and not flip_x)
          x_boundary = 1 + 2*shift_x + (0 if is_left else 2*k)
          y_boundary = 1 + 2*shift_y + (0 if is_top else 2*k)
          measure_boundary_filter = []
          for m in range(n_measure_dxd):
            if (use_rotated_z and measurement_sequence_measure_dxd[m][2][1]<y_boundary and measurement_sequence_measure_dxd[m][1]=='M:Z') \
              or (not use_rotated_z and measurement_sequence_measure_dxd[m][2][0]<x_boundary and measurement_sequence_measure_dxd[m][1]=='M:X'):
              measure_boundary_filter.append(m)
          measure_boundary_filter.extend([ idx_m + rr*n_measure_dxd for idx_m in measure_boundary_filter for rr in range(1, r)])
          measure_boundary_filter = np.array(measure_boundary_filter, dtype=idx_t)
          measure_boundary_filter = measure_boundary_filter.reshape(r, len(measure_boundary_filter)//r)
          cached_translation_idx_map.append(measure_boundary_filter)
        else:
          measure_boundary_filter = cached_translation_idx_map[idx_kernel_row]
        for rr in range(r):
          kernel_result_translation_map[idx_kernel_row, :, rr] = arrayops_sum(
              arrayops_gather(det_bits_dxd, measure_boundary_filter[rr], axis=1),
              axis=1
            ) % 2

      data_qubits_kxk = None
      measure_qubits_kxk = None
      measurement_sequence_measure_qubits_kxk = None
      measurement_sequence_data_qubits_kxk = None
      if make_cached_det_bits_map or make_cached_data_bits_map:
        data_qubits_kxk, measure_qubits_kxk = shift_frame(data_qubits_ord_kxk, measure_qubits_ord_kxk, k, d, shift_x, shift_y)
        measurement_sequence_measure_qubits_kxk = sorted(measure_qubits_kxk)
        measurement_sequence_data_qubits_kxk = sorted(data_qubits_kxk)

      # Find the index of the measure qubit in the dxd surface code through the position map
      det_bits_kxk_idx_filter = None
      if make_cached_det_bits_map:
        det_bits_kxk_idx_filter = np.zeros(r*n_measure_kxk, dtype=idx_t)
        for i in range(n_measure_kxk):
          pos_q_kxk = measurement_sequence_measure_qubits_kxk[i][2]
          found_match = False
          for j in range(n_measure_dxd):
            if np.array_equal(measurement_sequence_measure_dxd[j][2], pos_q_kxk):
              found_match = True
              for iround in range(r):
                det_bits_kxk_idx_filter[i + iround*n_measure_kxk] = j + iround*n_measure_dxd
              break
          if not found_match:
            raise RuntimeError(f"Measure qubit with position ({pos_q_kxk[0], pos_q_kxk[1]}) not found in the dxd surface code.")
        cached_det_bits_map.append(det_bits_kxk_idx_filter)
      else:
        det_bits_kxk_idx_filter = cached_det_bits_map[shift_x + shift_y*n_shifts]
      # Filter det_bits
      det_bits_kxk_all.append(arrayops_gather(det_bits_dxd, det_bits_kxk_idx_filter, axis=1))
      
      # Find the index of the data qubit on the dxd surface code through the position map
      if data_bits_dxd is not None:
        obs_bits_kxk_idx_filter = None
        data_bits_kxk_idx_filter = None
        if make_cached_data_bits_map:
          obs_bits_kxk_idx_filter = np.zeros(k, dtype=idx_t)
          data_bits_kxk_idx_filter = np.zeros(k**2, dtype=idx_t)
          iobs = 0
          for i in range(n_data_kxk):
            pos_q_kxk = measurement_sequence_data_qubits_kxk[i][2]
            lbl_kxk = measurement_sequence_data_qubits_kxk[i][1]
            found_match = False
            for j in range(n_data_dxd):
              if np.array_equal(measurement_sequence_data_dxd[j][2], pos_q_kxk):
                found_match = True
                data_bits_kxk_idx_filter[i] = j
                if ('XL' in lbl_kxk and not use_rotated_z) or ('ZL' in lbl_kxk and use_rotated_z):
                  obs_bits_kxk_idx_filter[iobs] = j
                  iobs = iobs + 1
                break
            if not found_match:
              raise RuntimeError(f"Data qubit with position ({pos_q_kxk[0], pos_q_kxk[1]}) not found in the dxd surface code.")
          # Reverse the order of the obs_bits filter to match stim conventions
          obs_bits_kxk_idx_filter = np.flip(obs_bits_kxk_idx_filter, axis=0)
          cached_data_bits_map.append(data_bits_kxk_idx_filter)
          cached_obs_bits_map.append(obs_bits_kxk_idx_filter)
        else:
          obs_bits_kxk_idx_filter = cached_obs_bits_map[shift_x + shift_y*n_shifts]
          data_bits_kxk_idx_filter = cached_data_bits_map[shift_x + shift_y*n_shifts]
        # Filter data_measurements
        data_bits_kxk_all.append(arrayops_gather(data_bits_dxd, data_bits_kxk_idx_filter, axis=1))
        obs_bits_kxk_all.append(arrayops_gather(data_bits_dxd, obs_bits_kxk_idx_filter, axis=1))

  if make_cached_det_bits_map or make_cached_data_bits_map or make_cached_translation_idx_map:
    dict_group_det_bits_kxk_[opts] = [cached_det_bits_map, cached_data_bits_map, cached_obs_bits_map, cached_translation_idx_map]

  for i in range(len(det_bits_kxk_all)):
    det_bits_kxk_all[i] = arrayops_cast(det_bits_kxk_all[i], binary_t)
  for i in range(len(data_bits_kxk_all)):
    data_bits_kxk_all[i] = arrayops_cast(data_bits_kxk_all[i], binary_t)
  for i in range(len(obs_bits_kxk_all)):
    obs_bits_kxk_all[i] = arrayops_cast(obs_bits_kxk_all[i], binary_t)

  return \
    arrayops_stack(det_bits_kxk_all, axis=0) if len(det_bits_kxk_all)>0 else None, \
    arrayops_stack(data_bits_kxk_all, axis=0) if len(data_bits_kxk_all)>0 else None, \
    arrayops_stack(obs_bits_kxk_all, axis=0) if len(obs_bits_kxk_all)>0 else None, \
    kernel_result_translation_map


def split_measurements(measurements, d, idx_t=np.int32):
  """
  split_measurements: Split the list of measurements from stim to
  detector bits (stabilizer measurements), observable bits (measurements that correspond to the defined observable),
  and data bits (measurements of all data qubits).
  Arguments:
  - measurements: All measurements of stabilizers and data qubits. Data qubit measurements come at the very end.
    The shape should be [number of samples, r*(d^2-1)+d^2], with r being the number of rounds, and d being the distance of the surface code.
  - d: Distance of the surface code
  - idx_t: Data type for the indices
  Returns:
  - Detector bits: 2D array of shape=[number of samples, r*(d^2-1)]
  - Observable bits: 2D array of shape=[number of samples, d]
  - Data bits: 2D array of shape=[number of samples, d^2]
  """
  n_measurements = idx_t(arrayops_shape(measurements, 1))
  # Measurements on data qubits come last
  exclude_indices = np.array([-x-1 for x in range(d**2)], dtype=idx_t)
  exclude_indices = exclude_indices + n_measurements
  # Out of all measurements on data qubits, the logical qubit measurements are those on the boundary of the lattice.
  # All other equivalent X_L/Z_L operators can be found through the combination of ancilla measurements and the chosen data qubits giving us the logical qubit.
  exclude_indices_obsL = np.array([-x-1 for x in range(d*(d-1), d**2)], dtype=idx_t)
  exclude_indices_obsL = exclude_indices_obsL + n_measurements

  det_bits = delete_elements(measurements, exclude_indices, axis=1)
  obs_bits = arrayops_gather(measurements, exclude_indices_obsL, axis=1)
  data_bits = arrayops_gather(measurements, exclude_indices, axis=1)

  # Reverse the order of data_bits because exclude_indices starts from the last data qubit measurement, not the first
  # This would conform with the ordering from stim.
  data_bits = flip_elements(data_bits, axis=1)

  return det_bits, obs_bits, data_bits


# Global dictionary to avoid reassembling the det_bits -> det_evts map.
dict_det_bits_to_det_evts_ = {}

def translate_det_bits_to_det_evts(d, r, k, obs_type, det_bits_kxk_all, final_det_evts):
  """
  translate_det_bits_to_det_evts: Translate the detector bits to detector events.
  Arguments:
  - obs_type: Type of observable to consider. Either "ZL" or "XL".
  - k: Size of the kernel
  - det_bits_kxk_all: Detector bits for the kxk kernel subsets of the surface code
  - final_det_evts: Final detector events for the surface code
  Returns:
  - Detector events for the kxk kernel subsets of the surface code
  """
  n_samples = arrayops_shape(det_bits_kxk_all, 1)
  n_shifts = d-k+1
  n_kernels = n_shifts**2
  na = k**2-1

  key = (obs_type, d, r, k)
  cached_map = dict_det_bits_to_det_evts_.get(key, None)
  make_cached_map = cached_map is None
  if make_cached_map:
    cached_map = [ None, [] ]

  det_bits_kxk_all_reshaped = arrayops_reshape(det_bits_kxk_all, (n_kernels, n_samples, r, na))
  det_evts_int = arrayops_reshape(
    arrayops_abs((det_bits_kxk_all_reshaped[:,:,1:,:] - det_bits_kxk_all_reshaped[:,:,:-1,:])),
    (n_kernels, n_samples, -1)
  )

  measure_qubits_ord_kxk = None
  measure_qubits_ord_dxd_filtered = None

  filter_kxk_pos_idxs = None
  if make_cached_map:
    # Find the index of the measure qubit in the dxd surface code through the position map
    measure_qubits_ord_dxd = get_measure_qubits_ord(d)
    measure_qubits_ord_dxd_filtered = []
    for qq in measure_qubits_ord_dxd:
      if (obs_type=="ZL" and qq[1]=="M:Z") or (obs_type=="XL" and qq[1]=="M:X"):
        measure_qubits_ord_dxd_filtered.append(qq[2])

    measure_qubits_ord_kxk = get_measure_qubits_ord(k)
    measure_qubits_ord_kxk_reord = sorted(measure_qubits_ord_kxk)

    filter_kxk_pos_idxs = []
    for rqq in measure_qubits_ord_kxk:
      for iqq, qq in enumerate(measure_qubits_ord_kxk_reord):
        if ((obs_type=="ZL" and qq[1]=="M:Z") or (obs_type=="XL" and qq[1]=="M:X")) and np.array_equal(rqq[2], qq[2]):
          filter_kxk_pos_idxs.append(iqq)
    cached_map[0] = filter_kxk_pos_idxs
  else:
    filter_kxk_pos_idxs = cached_map[0]
  det_bits_kxk_all_first = arrayops_cast(
    arrayops_reshape(
      arrayops_gather(det_bits_kxk_all_reshaped[:,:,0], filter_kxk_pos_idxs, axis=2),
      (n_kernels, n_samples, -1)
    ),
    det_bits_kxk_all.dtype
  )

  det_bits_kxk_all_last = []
  for shift_y in range(n_shifts):
    for shift_x in range(n_shifts):
      kernel_pos_map = None
      if make_cached_map:
        kernel_pos_map = []
        _, measure_ord_kxk_shifted = shift_frame(None, measure_qubits_ord_kxk, k, d, shift_x, shift_y)
        for iqq, qqk in enumerate(measure_ord_kxk_shifted):
          for jqq, qqd in enumerate(measure_qubits_ord_dxd_filtered):
            if np.array_equal(qqk[2], qqd):
              kernel_pos_map.append(jqq)
              break
        cached_map[1].append(kernel_pos_map)
      else:
        kernel_pos_map = cached_map[1][shift_x + shift_y*n_shifts]
      # Filter the kernel_pos_map to only include the ZL or XL observables
      det_bits_kxk_all_last.append(arrayops_gather(final_det_evts, kernel_pos_map, axis=1))
  det_bits_kxk_all_last = arrayops_cast(arrayops_stack(det_bits_kxk_all_last, axis=0), det_bits_kxk_all.dtype)

  if make_cached_map:
    dict_det_bits_to_det_evts_[key] = cached_map
  
  return arrayops_concatenate([det_bits_kxk_all_first, det_evts_int, det_bits_kxk_all_last], axis=2)


def decompose_state_from_bits(det_bits, r):
  """
  decompose_state_from_bits: Decompose the state of the surface code at each round from the detector bits (stabilizer measurements).
  
  This function provides a ternary representation with state=-1 meaning no error, 0 meaning a possible error, and 1 meaning a certain error.
  It collapses detector sequences of the form 010 or 101, so the two representations do not have 1-1 correspondence.
  Moreover, it may give different states to detector sequence palindromes, e.g., 1000 -> 0,0,0 vs. 0001 -> -1,-1,0.
  
  Consider also the following detector bit sequence: 0111000.
  - The corresponding state vector is 0,0,1,0,0.
  - The corresponding detector event sequence is 100100.
  If we were to naively interpret these sequences, we would think that there was a bit flip at the second and fifth rounds.
  However, another interpretation is that the first bit flip happened in the first round, which is not captured directly from detection events.
  The state vector in this case correctly describes the first state to be 0 (uncertain error assignment)
  and predicts correctly a return to the same state at the end, just like how detection events would also predict.

  Arguments:
  - det_bits: Detector bits. Shape should be consistent with [number of samples, r, d^2-1]
  - r: Number of rounds
  Returns:
  - 3D array of shape=[number of samples, r-2, d^2-1] for the state of the surface code at each round.
  """
  dt = det_bits.dtype
  det_bits_r = arrayops_reshape(det_bits, (arrayops_shape(det_bits, 0), r, -1))
  ns = arrayops_shape(det_bits_r, 0)
  nt = r
  ndet = arrayops_shape(det_bits_r, 2)
  state_tracker = -arrayops_ones(shape=(ns, ndet), dtype=dt)
  delta_tracker = arrayops_ones(shape=(ns, ndet), dtype=dt)
  states = []
  for t in range(nt-2):
    input = det_bits_r[:,t:t+3,:]
    data_err = input[:,0,:] + input[:,2,:] - input[:,0,:]*input[:,2,:]*2 # 001/011/110/100-like sequences
    measure_err = input[:,1,:]*(1 - (input[:,0,:] + input[:,2,:])) + input[:,0,:]*input[:,2,:] # 010/101-like sequences

    delta_tracker = delta_tracker*(1-measure_err*2)
    state_tracker = arrayops_maximum(
      arrayops_minimum(
        state_tracker + delta_tracker*data_err,
        arrayops_ones_like(state_tracker)
      ),
      -arrayops_ones_like(state_tracker)
    )
    delta_tracker = delta_tracker*(1-state_tracker*state_tracker*(1-measure_err)) - state_tracker*(1-measure_err)

    states.append(deepcopy(state_tracker))
  return arrayops_stack(states, axis=1)
