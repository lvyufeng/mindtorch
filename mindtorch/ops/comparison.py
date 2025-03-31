"""comparison op"""
import numpy as np

from mindtorch.executor import execute

# allclose
def allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    return np.allclose(input.numpy(), other.numpy(), rtol, atol, equal_nan)

# argsort
def argsort(input, dim=-1, descending=False, stable=False):
    return sort(input, dim=dim, descending=descending, stable=stable)[1]

# eq
def eq(input, other):
    return execute('equal', input, other)

# equal
def equal(input, other):
    return execute('equal', input, other)

# ge
def ge(input, other):
    return execute('greater_equal', input, other)

# gt
def gt(input, other):
    return execute('greater', input, other)

# greater
def greater(input, other):
    return gt(input, other)

# isclose
def isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    return execute('isclose', input, other, rtol, atol, equal_nan)

# isfinite
def isfinite(input):
    return execute('isfinite', input)

# isin

# isinf
def isinf(input):
    return execute('isinf', input)

# isposinf

# isneginf

# isnan
def isnan(input):
    return execute('not_equal', input, input)

# isreal

# kthvalue

# le
def le(input, other):
    return execute('less_equal', input, other)

# less_equal
def less_equal(input, other):
    return le(input, other)

# lt
def lt(input, other):
    return execute('less', input, other)

# less
def less(input, other):
    return lt(input, other)

# maximum
def maximum(input, other):
    return execute('maximum', input, other)

# minimum
def minimum(input, other):
    return execute('minimum', input, other)

# fmax

# fmin

# ne
def ne(input, other):
    return execute('not_equal', input, other)

# not_equal
def not_equal(input, other):
    return ne(input, other)

# sort
def sort(input, *, dim=-1, descending=False, stable=False):
    return execute('sort_ext', input, dim, descending, stable)

# topk
def topk(input, k, dim=-1, largest=True, sorted=True):
    return execute('topk_ext', input, k, dim, largest, sorted)

# msort
def msort(input):
    return sort(input, dim=0)

__all__ = [
    'allclose',
    'argsort',
    'eq',
    'equal',
    'ge',
    'gt',
    'greater',
    'isclose',
    'isfinite',
    'isinf',
    # isposinf,
    # isneginf,
    'isnan',
    # isreal,
    # kthvalue,
    'le',
    'less_equal',
    'lt',
    'less',
    'maximum',
    'minimum',
    'ne',
    'not_equal',
    'sort',
    'topk',
    'msort',
]
