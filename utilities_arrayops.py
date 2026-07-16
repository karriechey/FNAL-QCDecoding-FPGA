import numpy as np
import tensorflow as tf


def convert_to_tfdtype(dtype):
  """
  convert_to_tfdtype: Convert a numpy data type to a TensorFlow data type.
  Arguments:
  - dtype: Data type to convert.
  """
  if dtype is None or isinstance(dtype, tf.DType):
    return dtype

  if dtype == np.int8:
    return tf.int8
  elif dtype == np.int16:
    return tf.int16
  elif dtype == np.int32:
    return tf.int32
  elif dtype == np.int64:
    return tf.int64

  elif dtype == np.uint8:
    return tf.uint8
  elif dtype == np.uint16:
    return tf.uint16
  elif dtype == np.uint32:
    return tf.uint32
  elif dtype == np.uint64:
    return tf.uint64
  
  elif dtype == np.float16:
    return tf.float16
  elif dtype == np.float32:
    return tf.float32
  elif dtype == np.float64:
    return tf.float64
  else:
    raise ValueError("Could not recognize the data type.")


def convert_to_npdtype(dtype):
  """
  convert_to_npdtype: Convert a TensorFlow data type to a numpy data type.
  Arguments:
  - dtype: Data type to convert.
  """
  if dtype is None or isinstance(dtype, np.dtype):
    return dtype

  # Handle TF DTypes first. tf.float32 == np.float32 evaluates True, so a
  # tf.DType would otherwise fall into the numpy-type-class branch below and
  # np.dtype(tf.float32) raises. Resolving TF dtypes here avoids that.
  if isinstance(dtype, tf.DType):
    return dtype.as_numpy_dtype

  # Handle numpy type classes (e.g., np.int8, np.float32)
  if dtype in (np.int8, np.int16, np.int32, np.int64,
               np.uint8, np.uint16, np.uint32, np.uint64,
               np.float16, np.float32, np.float64, np.bool_):
    return np.dtype(dtype)

  if dtype == tf.int8:
    return np.int8
  elif dtype == tf.int16:
    return np.int16
  elif dtype == tf.int32:
    return np.int32
  elif dtype == tf.int64:
    return np.int64

  elif dtype == tf.uint8:
    return np.uint8
  elif dtype == tf.uint16:
    return np.uint16
  elif dtype == tf.uint32:
    return np.uint32
  elif dtype == tf.uint64:
    return np.uint64
  
  elif dtype == tf.float16:
    return np.float16
  elif dtype == tf.float32:
    return np.float32
  elif dtype == tf.float64:
    return np.float64
  else:
    raise ValueError("Could not recognize the data type.")


def arrayops_shape(array, axis = None):
  """
  arrayops_shape: Get the shape of an array/tensor.
  Arguments:
  - array: Array in which elements are to be flipped.
  - axis: Axis along which to flip elements.
  """
  res = None
  if isinstance(array, np.ndarray):
    res = array.shape
  else:
    res = tf.shape(array)
  if axis is not None:
    return res[axis]
  else:
    return res


def arrayops_rank(array):
  """
  arrayops_shape: Get the rank of an array/tensor.
  Arguments:
  - array: Array in which elements are to be flipped.
  """
  res = None
  if isinstance(array, np.ndarray):
    return len(array.shape)
  else:
    return tf.rank(array)


def arrayops_range(n, dtype, step = 1):
  """
  arrayops_range: Get an array of elements in a range
  Arguments:
  - n: Span of the range
  """
  if isinstance(dtype, np.dtype):
    return np.arange(n, step = step, dtype = dtype)
  else:
    return tf.range(n, delta = step, dtype = dtype)


def arrayops_setdiff1d(a, b):
  """
  arrayops_setdiff1d: Get an array of elements that includes all those in one except another.
  Arguments:
  - a: Array to include
  - b: Array to exclude
  """
  if isinstance(a, np.ndarray):
    return np.setdiff1d(a, b)
  else:
    return tf.compat.v1.setdiff1d(a, b)


def delete_elements(array, indices, axis):
  """
  delete_elements: Delete elements from an array.
  Arguments:
  - array: Array from which elements are to be deleted.
  - indices: Indices of elements to be deleted.
  - axis: Axis along which to delete elements.
  """
  if isinstance(array, np.ndarray):
    return np.delete(array, indices, axis=axis)
  else:
    gather_indices = arrayops_setdiff1d(arrayops_range(arrayops_shape(array, axis)), indices)
    return tf.gather(array, gather_indices, axis=axis)


def flip_elements(array, axis):
  """
  flip_elements: Flip elements in an array.
  Arguments:
  - array: Array in which elements are to be flipped.
  - axis: Axis along which to flip elements.
  """
  if isinstance(array, np.ndarray):
    return np.flip(array, axis=axis)
  else:
    return tf.reverse(array, axis=[axis])


def make_const_array_like(array, like_array, assume_same_dtype=False, dtype=None):
  """
  make_const_array_like: Make a constant array like another array.
  Arguments:
  - array: Array to be made like another array.
  - like_array: Array to be made like.
  - assume_same_dtype: Assume the same data type as the like array.
  - dtype: Data type of the array if not assuming the same data type.
  """
  if isinstance(like_array, np.ndarray):
    return np.array(array, dtype=like_array.dtype if assume_same_dtype else convert_to_npdtype(dtype))
  else:
    return tf.constant(array, dtype=like_array.dtype if assume_same_dtype else convert_to_tfdtype(dtype))


def arrayops_zeros(shape, dtype):
  """
  arrayops_zeros: Create an array of zeros.
  Arguments:
  - shape: Shape of the array.
  - dtype: Data type of the array.
  """
  if isinstance(dtype, np.dtype):
    return np.zeros(shape, dtype=dtype)
  elif isinstance(dtype, tf.DType):
    return tf.zeros(shape, dtype=dtype)
  else:
    raise ValueError("Could not recognize the tensor type from the data type.")


def arrayops_ones(shape, dtype):
  """
  arrayops_ones: Create an array of ones.
  Arguments:
  - shape: Shape of the array.
  - dtype: Data type of the array.
  """
  if isinstance(dtype, np.dtype):
    return np.ones(shape, dtype=dtype)
  elif isinstance(dtype, tf.DType):
    return tf.ones(shape, dtype=dtype)
  else:
    raise ValueError("Could not recognize the tensor type from the data type.")


def arrayops_zeros_like(array):
  """
  arrayops_zeros_like: Create an array of zeros like another array.
  Arguments:
  - array: Array to be made like.
  """
  if isinstance(array, np.ndarray):
    return np.zeros_like(array)
  else:
    return tf.zeros_like(array)


def arrayops_ones_like(array):
  """
  arrayops_ones_like: Create an array of ones like another array.
  Arguments:
  - array: Array to be made like.
  """
  if isinstance(array, np.ndarray):
    return np.ones_like(array)
  else:
    return tf.ones_like(array)


def arrayops_abs(array):
  """
  arrops_abs: Absolute value of an array.
  Arguments:
  - array: Array of which to take the absolute value.
  """
  if isinstance(array, np.ndarray):
    return np.abs(array)
  else:
    return tf.math.abs(array)


def arrayops_sign(array):
  """
  arrayops_sign: Sign of an array.
  Arguments:
  - array: Array of which to take the sign.
  """
  if isinstance(array, np.ndarray):
    return np.sign(array)
  else:
    return tf.math.sign(array)


def arrayops_sum(array, axis=None, keepdims=False):
  """
  arrayops_sum: Sum of an array along an axis.
  The operation is equivalent to np.sum or tf.reduce_sum.
  Arguments:
  - array: Array of which to take the sum.
  - axis: Axis along which to take the sum.
  - keepdims: Whether to keep the dimensions of the array.
  """
  if isinstance(array, np.ndarray):
    return np.sum(array, axis=axis, keepdims=keepdims)
  else:
    return tf.reduce_sum(array, axis=axis, keepdims=keepdims)


def arrayops_mean(array, axis=None, keepdims=False):
  """
  arrayops_mean: Mean of an array along an axis.
  The operation is equivalent to np.mean or tf.reduce_mean.
  Arguments:
  - array: Array of which to take the mean.
  - axis: Axis along which to take the mean.
  - keepdims: Whether to keep the dimensions of the array.
  """
  if isinstance(array, np.ndarray):
    return np.mean(array, axis=axis, keepdims=keepdims)
  else:
    return tf.reduce_mean(array, axis=axis, keepdims=keepdims)


def arrayops_prod(array, axis=None, keepdims=False):
  """
  arrayops_prod: Product of an array along an axis.
  The operation is equivalent to np.prod or tf.reduce_prod.
  Arguments:
  - array: Array of which to take the product.
  - axis: Axis along which to take the product.
  - keepdims: Whether to keep the dimensions of the array.
  """
  if isinstance(array, np.ndarray):
    return np.prod(array, axis=axis, keepdims=keepdims)
  else:
    return tf.reduce_prod(array, axis=axis, keepdims=keepdims)
  

def arrayops_max(array, axis=None, keepdims=False):
  """
  arrayops_max: Maximum value of an array along an axis.
  The operation is equivalent to np.max or tf.reduce_max.
  Arguments:
  - array: Array of which to take the maximum value.
  - axis: Axis along which to take the maximum value.
  - keepdims: Whether to keep the dimensions of the array.
  """
  if isinstance(array, np.ndarray):
    return np.max(array, axis=axis, keepdims=keepdims)
  else:
    return tf.reduce_max(array, axis=axis, keepdims=keepdims)


def arrayops_min(array, axis=None, keepdims=False):
  """
  arrayops_min: Minimum value of an array along an axis.
  The operation is equivalent to np.min or tf.reduce_min.
  Arguments:
  - array: Array of which to take the minimum value.
  - axis: Axis along which to take the minimum value.
  - keepdims: Whether to keep the dimensions of the array.
  """
  if isinstance(array, np.ndarray):
    return np.min(array, axis=axis, keepdims=keepdims)
  else:
    return tf.reduce_min(array, axis=axis, keepdims=keepdims)


def arrayops_logsumexp(array, axis=None, keepdims=False):
  """
  arrayops_logsumexp: Logarithm of the sum of exponentials of an array along an axis.
  The operation is equivalent to np.logsumexp or tf.reduce_logsumexp.
  Arguments:
  - array: Array of which to take the logarithm of the sum of exponentials.
  - axis: Axis along which to take the logarithm of the sum of exponentials.
  - keepdims: Whether to keep the dimensions of the array.
  """
  if isinstance(array, np.ndarray):
    return np.log(np.sum(np.exp(array), axis=axis, keepdims=keepdims))
  else:
    return tf.reduce_logsumexp(array, axis=axis, keepdims=keepdims)


def arrayops_any(array, axis=None, keepdims=False):
  """
  arrayops_any: Logical OR of an array along an axis.
  The operation is equivalent to np.any or tf.reduce_any.
  Arguments:
  - array: Array of which to take the logical OR.
  - axis: Axis along which to take the logical OR.
  - keepdims: Whether to keep the dimensions of the array.
  """
  if isinstance(array, np.ndarray):
    return np.any(array, axis=axis, keepdims=keepdims)
  else:
    return tf.reduce_any(array, axis=axis, keepdims=keepdims)


def arrayops_all(array, axis=None, keepdims=False):
  """
  arrayops_all: Logical AND of an array along an axis.
  The operation is equivalent to np.all or tf.reduce_all.
  Arguments:
  - array: Array of which to take the logical AND.
  - axis: Axis along which to take the logical AND.
  - keepdims: Whether to keep the dimensions of the array.
  """
  if isinstance(array, np.ndarray):
    return np.all(array, axis=axis, keepdims=keepdims)
  else:
    return tf.reduce_all(array, axis=axis, keepdims=keepdims)


def arrayops_maximum(arr1, arr2):
  """
  arrayops_maximum: Element-wise maximum value.
  Arguments:
  - arr1: First array.
  - arr2: Second array.
  """
  if isinstance(arr1, np.ndarray):
    return np.maximum(arr1, arr2)
  else:
    return tf.maximum(arr1, arr2)


def arrayops_minimum(arr1, arr2):
  """
  arrayops_minimum: Element-wise minimum value.
  Arguments:
  - arr1: First array.
  - arr2: Second array.
  """
  if isinstance(arr1, np.ndarray):
    return np.minimum(arr1, arr2)
  else:
    return tf.minimum(arr1, arr2)


def arrayops_concatenate(arrays, axis):
  """
  arrayops_concatenate: Concatenate arrays.
  Arguments:
  - arrays: Arrays to concatenate.
  - axis: Axis along which to concatenate.
  """
  if isinstance(arrays[0], np.ndarray):
    return np.concatenate(arrays, axis=np.int32(axis))
  else:
    return tf.concat(arrays, axis=int(axis))


def arrayops_reshape(array, shape):
  """
  arrayops_reshape: Reshape an array.
  Arguments:
  - array: Array to reshape.
  - shape: Shape of the reshaped array.
  """
  if isinstance(array, np.ndarray):
    return np.reshape(array, shape)
  else:
    return tf.reshape(array, shape)


def arrayops_swapaxes(array, axis1 = None, axis2 = None, perm = None):
  """
  arrayops_swapaxes: Swap axes of an array.
  Arguments:
  - array: Array of which to swap axes.
  - axis1: First axis to swap.
  - axis2: Second axis to swap.
  - perm: Permutation of axes as would be given to tf.transpose.
  Note in the arguments that either axis1 and axis2, or perm can be None, never both.
  """
  if axis1 is None and axis2 is None and perm is None:
    raise ValueError("axis1, axis2, and perm cannot be all None.")
  if (axis1 is not None and axis2 is None) or (axis2 is not None and axis1 is None):
    raise ValueError("axis1 and axis2 should both be set if set at all.")
  if isinstance(array, np.ndarray):
    if axis1 is None and axis2 is None:
      raise ValueError("numpy implementation of swapaxes needs axis1 and axis2.")
    return np.swapaxes(array, axis1, axis2)
  else:
    p = None
    if perm is not None:
      p = perm
    else:
      p = [(i if (i!=axis1 and i!=axis2) else axis2 if i==axis1 else axis1) for i in range(arrayops_rank(array))]
    return tf.transpose(array, perm=p)


def arrayops_gather(array, indices, axis):
  """
  arrayops_gather: Gather elements from an array.
  Arguments:
  - array: Array from which to gather elements.
  - indices: Indices of elements to gather.
  - axis: Axis along which to gather elements.
  """
  if isinstance(array, np.ndarray):
    return np.take(array, indices, axis=np.int32(axis))
  else:
    return tf.gather(array, tf.cast(indices, tf.int32), axis=int(axis))


def arrayops_gather_nd(array, indices, batch_dims=0):
  """
  arrayops_gather_nd: Gather elements from an array using N-dimensional indices.
  Arguments:
  - array: Array from which to gather elements.
  - indices: N-dimensional indices of elements to gather.
  - batch_dims: Number of batch dimensions, only works with TensorFlow for now.
  """
  if isinstance(array, np.ndarray):
    return array[tuple(indices)]
  else:
    return tf.gather_nd(array, tf.cast(indices, tf.int32), batch_dims=batch_dims)
  

def arrayops_stack(arrays, axis):
  """
  arrayops_stack: Stack arrays.
  Arguments:
  - arrays: Arrays to stack.
  - axis: Axis along which to stack.
  """
  if isinstance(arrays[0], np.ndarray):
    return np.stack(arrays, axis=np.int32(axis))
  else:
    return tf.stack(arrays, axis=int(axis))


def arrayops_cast(array, dtype):
  """
  arrayops_cast: Cast an array to a different data type.
  Arguments:
  - array: Input array.
  - dtype: Data type of the cast.
  """
  if isinstance(array, np.ndarray):
    return array.astype(convert_to_npdtype(dtype))
  else:
    return tf.cast(array, dtype=convert_to_tfdtype(dtype))



class VariableBounds:
  bound_zlike = 12


  @staticmethod
  def set_bound_zlike(bound):
    if bound<0:
      bound = -bound
    VariableBounds.bound_zlike = bound
  

  @staticmethod
  def get_bound_zlike(dtype):
    return tf.cast(VariableBounds.bound_zlike, dtype)
  

  @staticmethod
  def clip_zlike(zlike):
    zb = VariableBounds.get_bound_zlike(zlike.dtype)
    return tf.clip_by_value(zlike, -zb, zb)


  @staticmethod
  def clip_frac(frac):
    zb = VariableBounds.get_bound_zlike(frac.dtype)
    ez = tf.math.exp(zb)
    fz = ez/(ez+1)
    return tf.clip_by_value(frac, -fz+1, fz)


  @staticmethod
  def clip_exp(expv):
    zb = VariableBounds.get_bound_zlike(expv.dtype)
    ez = tf.math.exp(zb)
    return tf.clip_by_value(expv, 1/ez, ez)
  

  @staticmethod
  def clip_prob(prob):
    return VariableBounds.clip_frac(prob)
