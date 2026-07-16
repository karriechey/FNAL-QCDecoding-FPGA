import numpy as np


def get_types(d, r, k):
  """
  get_types: Get global types for execution.
  Arguments:
  - d: (Maximum) distance of the surface code
  - r: (Maximum) number of rounds
  - k: Distance of the kernel
  """
  binary_t = np.int8 # Could use even less if numpy allowed
  time_t = None
  idx_t = None
  packed_t = None

  n_all_measurements = r*(d**2-1) + d**2

  if d<=8:
    pass
  elif d>8 and d<=16:
    packed_t = np.int16
  elif d>16 and d<=32:
    packed_t = np.int32
  elif d>32 and d<=64:
    packed_t = np.int64
  elif d>64 and d<=128:
    packed_t = np.int128
  elif d>128 and d<=256:
    packed_t = np.int256
  else:
    raise ValueError("d is too large to set packed_t.")
  
  idx_t = np.int8
  if n_all_measurements > np.iinfo(idx_t).max:
    idx_t = np.int16
  if n_all_measurements > np.iinfo(idx_t).max:
    idx_t = np.int32
  if n_all_measurements > np.iinfo(idx_t).max:
    idx_t = np.int64
  if n_all_measurements > np.iinfo(idx_t).max:
    idx_t = np.int128
  if n_all_measurements > np.iinfo(idx_t).max:
    idx_t = np.int256
  if n_all_measurements > np.iinfo(idx_t).max:
    raise ValueError(f"n_all_measurements = {n_all_measurements} is too large for idx_t.")
  
  time_t = np.int8
  if r > np.iinfo(time_t).max:
    time_t = np.int16
  if r > np.iinfo(time_t).max:
    time_t = np.int32
  if r > np.iinfo(time_t).max:
    time_t = np.int64
  if r > np.iinfo(time_t).max:
    time_t = np.int128
  if r > np.iinfo(time_t).max:
    time_t = np.int256
  if r > np.iinfo(time_t).max:
    raise ValueError(f"r = {r} is too large for time_t.")
  
  del n_all_measurements
  return binary_t, time_t, idx_t, packed_t
