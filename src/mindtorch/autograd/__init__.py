"""autograd"""
from mindspore import Tensor as Variable
from .grad_mode import no_grad, enable_grad, inference_mode
from .function import value_and_grad, grad, Function, vjp
from .variable import Variable
from .tape import GradientTape