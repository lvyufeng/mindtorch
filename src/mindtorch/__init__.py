# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""core module"""
import os
import platform
from typing import (
    Any as _Any,
    Callable as _Callable,
    get_origin as _get_origin,
    Optional as _Optional,
    overload as _overload,
    TYPE_CHECKING,
    TypeVar as _TypeVar,
    Union as _Union,
)


from mindspore import context
from mindspore._c_expression import MSContext # pylint: disable=no-name-in-module, import-error

os.environ['RANK'] = os.getenv('RANK_ID', '0')
os.environ['WORLD_SIZE'] = os.getenv('MS_WORKER_NUM', '1')

if 'RANK_TABLE_FILE' in os.environ:
    del os.environ['RANK_TABLE_FILE']
DEVICE_TARGET = os.environ.get('DEVICE_TARGET', None)

if DEVICE_TARGET is not None and DEVICE_TARGET in ('CPU', 'GPU', 'Ascend'):
    context.set_context(device_target=DEVICE_TARGET)

if platform.system().lower() == 'linux':
    SOC = MSContext.get_instance().get_ascend_soc_version()
    if ('910b' not in SOC and '310' not in SOC):
        os.environ["MS_ALLOC_CONF"] = 'enable_vmm:True,vmm_align_size:2MB'

    if SOC in ('ascend910', 'ascend310b'):
        context.set_context(ascend_config={"precision_mode": "allow_mix_precision"})

strided = None
contiguous_format = None
preserve_format = None

inf = float("inf")
nan = float("nan")

from mindspore import default_generator, Generator
from mindspore.hal import Stream
from mindspore import multiprocessing
from mindspore.common.api import _pynative_executor

# fake pybind impl
from ._dtype import *

from ._tensor import Tensor, tensor, is_tensor, \
    FloatTensor, HalfTensor, BFloat16Tensor, LongTensor, DoubleTensor, IntTensor, \
    BoolTensor, ByteTensor

from . import _C, _jit_internal, _ops, version, export
from ._C.size import Size
from .types import device
from .ops import *
from mindtorch.amp import autocast, GradScaler
from mindtorch import (
    __future__ as __future__,
    amp as amp,
    random as random,
    serialization as serialization,
    utils as utils,
    jit as jit,
    fx as fx,
    compiler as compiler,
    overrides as overrides,
    profiler as profiler
)
from mindtorch.random import get_rng_state, initial_seed, manual_seed, seed, set_rng_state
from mindtorch.serialization import load, save
from . import optim, ops, nn, distributions, cuda, npu, distributed#, multiprocessing
from .autograd import no_grad, enable_grad, value_and_grad, inference_mode
from ._bind import get_default_dtype, set_default_dtype, set_default_device, get_default_device, finfo
from .storage import TypedStorage, UntypedStorage, Storage
from ._lowrank import svd_lowrank

AUTO_CAST_DTYE = {
    'cuda': bfloat16,
    'cpu': bfloat16,
    'npu': float16
}

def set_autocast_dtype(device_type, dtype):
    assert device_type in AUTO_CAST_DTYE.keys(), f'{device_type} is not in {AUTO_CAST_DTYE.keys()}'
    AUTO_CAST_DTYE[device_type] = dtype

def get_autocast_dtype(device_type):
    return AUTO_CAST_DTYE[device_type]

def is_autocast_enabled(device_type):
    return False

def use_deterministic_algorithms(flag: bool):
    context.set_context(deterministic='ON' if flag else 'OFF')

def is_grad_enabled():
    return _pynative_executor.enable_grad()

def compile(fn=None, *args, **kwargs):
    def wrap_func(fn):
        return fn
    if fn is not None:
        return wrap_func(fn)
    return wrap_func

def _has_compatible_shallow_copy_type(tensor, other):
    """
    Mimics the behavior of mindtorch._has_compatible_shallow_copy_type.

    Args:
        tensor (mindtorch.Tensor): The source tensor.
        other (mindtorch.Tensor): The target tensor to check compatibility.

    Returns:
        bool: True if `tensor` and `other` have compatible types for shallow copy.
    """
    # Check if both tensors have the same type
    if not is_tensor(tensor) or not is_tensor(other):
        return False

    # Check if both tensors are on the same device
    if tensor.shape != other.shape:
        return False

    # Compatibility confirmed
    return True

def typename(obj: _Any, /) -> str:
    """
    String representation of the type of an object.

    This function returns a fully qualified string representation of an object's type.
    Args:
        obj (object): The object whose type to represent
    Returns:
        str: the type of the object `o`
    Example:
        >>> x = torch.tensor([1, 2, 3])
        >>> torch.typename(x)
        'torch.LongTensor'
        >>> torch.typename(torch.nn.Parameter)
        'torch.nn.parameter.Parameter'
    """
    if isinstance(obj, Tensor):
        return obj.type()

    module = getattr(obj, "__module__", "") or ""
    qualname = ""

    if hasattr(obj, "__qualname__"):
        qualname = obj.__qualname__
    elif hasattr(obj, "__name__"):
        qualname = obj.__name__
    else:
        module = obj.__class__.__module__ or ""
        qualname = obj.__class__.__qualname__

    if module in {"", "builtins"}:
        return qualname
    return f"{module}.{qualname}"

__version__ = "0.0.1"
