import mindspore
from mindspore import context
BACKEND = context.get_context('device_target')

from .dtype import *
from .config import no_grad
from ._tensor import Tensor, setup_tensor, tensor, is_tensor
from ._functions import ones, zeros, flatten, uniform
from mindtorch import cuda as cuda

setup_tensor()

def manual_seed(seed):
    mindspore.set_seed(seed)

def device(device):
    return device

