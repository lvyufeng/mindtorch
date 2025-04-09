"""reduction op"""
from mindspore.ops.auto_generate.gen_arg_handler import dtype_to_type_id

import mindtorch
from mindtorch.executor import execute

# argmax
def argmax(input, dim=None, keepdim=False):
    return execute('argmax_ext', input, dim, keepdim)

# argmin
def argmin(input, dim=None, keepdim=False):
    return execute('argmin_ext', input, dim, keepdim)

# amax
def amax(input, dim, keepdim=False):
    return execute('reduce_max', input, dim, keepdim)

# amin
def amin(input, dim, keepdim=False):
    return execute('reduce_min', input, dim, keepdim)

# aminmax
def aminmax(input, *, dim=None, keepdim=False):
    if dim is None:
        dim = ()
    return amin(input, dim, keepdim), amax(input, dim, keepdim)

# all
def all(input, dim=None, keepdim=False, *, dtype=None):
    if dim is None:
        dim = ()
    return execute('reduce_all', input, dim, keepdim)

# any
def any(input, dim=None, keepdim=False):
    if dim is None:
        dim = ()
    return execute('reduce_any', input, dim, keepdim)

# max
def max(input, dim=None, keepdim=False, *, out=None):
    if dim is None and not keepdim:
        return execute('max', input)
    if mindtorch.is_tensor(dim):
        return mindtorch.maximum(input, dim)
    indices, output = execute('argmax_with_value', input, dim, keepdim)
    if out is None:
        return output, indices
    out[0].data = output
    out[1].data = indices
    return out

# min
def min(input, dim=None, keepdim=False, *, out=None):
    if dim is None and not keepdim:
        return execute('min', input)
    if mindtorch.is_tensor(dim):
        return mindtorch.minimum(input, dim)
    indices, output = execute('argmin_with_value', input, dim, keepdim)
    if out is None:
        return output, indices
    out[0].data = output
    out[1].data = indices
    return out

# dist

# logsumexp
def logsumexp(input, dim, keepdim=False):
    return execute('logsumexp', input, dim, keepdim)

# mean
def mean(input, dim=None, keepdim=False, *, dtype=None):
    return execute('mean_ext', input, dim, keepdim,
                   dtype if dtype is None else dtype_to_type_id('MeanExt', 'dtype', dtype))

# nanmean


# median
def median(input, dim=-1, keepdim=False):
    if dim is None:
        return execute('median_ext', input)
    return execute('median_dim', input, dim, keepdim)

# nanmedian


# mode


# norm
def vector_norm_ext(input, p=2, dim=None, keepdim=False, *, dtype=None):
    if float(p) in [0.0, 1.0, 2.0, 3.0]:
        return execute('linalg_vector_norm', input, float(p), dim, keepdim,
                        dtype if dtype is None else dtype_to_type_id('LinalgVectorNorm', 'dtype', dtype))
    if input.dtype in [mindtorch.bfloat16, mindtorch.float16, mindtorch.float32]:
        if dtype is None:
            return execute('lp_norm_v2', input, p, dim, keepdim, 0.0)
        return execute('lp_norm_v2', input, p, dim, keepdim, 0.0).to(dtype)

    cast_dtype = input.dtype if dtype is None else dtype
    input = input.to(mindtorch.float32)
    return execute('lp_norm_v2', input, p, dim, keepdim, 0.0).to(cast_dtype)

def matrix_norm_ext(A, ord='fro', dim=(-2, -1), keepdim=False, *, dtype=None):
    ndim = A.ndim
    row_axis, col_axis = _check_matrix_norm_axis(dim, ndim)
    _check_matrix_norm_ord(ord)
    if ord == 'fro':
        return vector_norm_ext(A, 2, dim, keepdim, dtype=dtype)
    if ord == 'nuc':
        res = _multi_svd_norm(A, row_axis, col_axis, 'sum')
        return _reshape_matrix_norm(A, res, dim, keepdim)
    if ord == 2:
        res = _multi_svd_norm(A, row_axis, col_axis, 'amax')
        return _reshape_matrix_norm(A, res, dim, keepdim)
    if ord == -2:
        res = _multi_svd_norm(A, row_axis, col_axis, 'amin')
        return _reshape_matrix_norm(A, res, dim, keepdim)
    if ord in [float('inf'), -float('inf')]:
        row_axis, col_axis = col_axis, row_axis
    if not keepdim and col_axis > row_axis:
        col_axis -= 1
    if ord < 0:
        return amin(vector_norm_ext(A, 1, row_axis, keepdim, dtype=dtype), col_axis, keepdim)
    return amax(vector_norm_ext(A, 1, row_axis, keepdim, dtype=dtype), col_axis, keepdim)

def norm(input, p='fro', dim=None, keepdim=False, dtype=None):
    if not isinstance(input, mindtorch.Tensor):
        raise TypeError(f"For `norm_ext`, the `input` must be Tensor!, but get {type(input)}.")
    if isinstance(p, (bool, int, float)):
        return vector_norm_ext(input, p, dim, keepdim, dtype=dtype)
    if p == 'fro':
        if isinstance(dim, (list, tuple)) and len(dim) > 2:
            raise ValueError(f"For `norm_ext`, the size of `dim` cannot be greater than 2 "
                             f"when the norm mode is `fro`.")
        return execute('linalg_vector_norm', input, 2.0, dim, keepdim,
                       dtype if dtype is None else dtype_to_type_id('LinalgVectorNorm', 'dtype', dtype))
    if p == 'nuc':
        dim = tuple(range(input.ndim)) if dim is None else dim
        return matrix_norm_ext(input, p, dim, keepdim, dtype=dtype)
    raise ValueError(f"For `norm_ext`, the value of `p` must be one of [int, float, inf, -inf, 'fro', 'nuc',] "
                     f"but got `{p}`.")

# nansum
def nansum(input, dim=None, keepdim=False, *, dtype=None):
    return execute('nansum', input, dim, keepdim,
                   dtype if dtype is None else dtype_to_type_id('Nansum', 'dtype', dtype))

# prod
def prod(input, dim=None, keepdim=False, *, dtype=None):
    return execute('prod_ext', input, dim, keepdim,
                   dtype if dtype is None else dtype_to_type_id('ProdExt', 'dtype', dtype))

# quantile

# nanquantile

# std
def std(input, dim=None, *, correction=1, keepdim=False):
    return execute('std', input, dim, correction, keepdim)

# std_mean
def std_mean(input, dim=None, *, correction=1, keepdim=False):
    return execute('std_mean', input, dim, correction, keepdim)

# sum
def sum(input, dim=None, keepdim=False, *, dtype=None):
    if 0 in input.shape:
        return mindtorch.tensor(0, dtype=dtype, device=input.device)
    return execute('sum_ext', input, dim, keepdim,
                   dtype if dtype is None else dtype_to_type_id('SumExt', 'dtype', dtype))

# unique
def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    if dim is None:
        y, inverse, counts = execute('unique2',
            input, sorted, return_inverse, return_counts)
    else:
        y, inverse, counts = execute('unique_dim', input, sorted, return_inverse, dim)
    if return_inverse and return_counts:
        return y, inverse, counts
    if return_inverse:
        return y, inverse
    if return_counts:
        return y, counts
    return y

# unique_consecutive
def unique_consecutive(input, return_inverse=False, return_counts=False, dim=None):
    output, idx, counts = execute('unique_consecutive', input, return_inverse, return_counts, dim)
    if return_inverse and return_counts:
        return output, idx, counts
    if return_inverse:
        return output, idx
    if return_counts:
        return output, counts
    return output

# var
def var(input, dim=None, *, correction=1, keepdim=False):
    return execute('var', input, dim, correction, keepdim)

# var_mean
def var_mean(input, dim=None, *, correction=1, keepdim=False):
    return execute('var_mean', input, dim, correction, keepdim)

# count_nonzero
def count_nonzero(input, dim=None):
    return execute('count_nonzero', input, dim)

__all__ = ['all', 'amax', 'amin', 'aminmax', 'any', 'argmax', 'argmin', 'count_nonzero',
           'logsumexp', 'max', 'mean', 'median', 'min', 'nansum',
           'norm', 'prod', 'std', 'std_mean', 'sum', 'unique', 'unique_consecutive',
           'var', 'var_mean']
 