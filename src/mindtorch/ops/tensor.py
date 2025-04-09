"""tensor op"""
import mindtorch
from mindspore._c_expression import typing # pylint: disable=no-name-in-module, import-error

def is_floating_point(input):
    return isinstance(input.dtype, typing.Float)

def is_tensor(input):
    return type(input) == mindtorch.Tensor or type(input) == mindtorch.nn.Parameter

def numel(input):
    return input.numel()

def as_tensor(data, dtype=None, device=None):
    return mindtorch.Tensor(data, dtype, device=device)

__all__ = ['as_tensor', 'is_floating_point', 'is_tensor', 'numel']
