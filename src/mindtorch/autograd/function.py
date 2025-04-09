"""functional autograd"""
from collections.abc import Generator

import functools
import mindspore
from mindspore.ops.composite import GradOperation
from mindspore.ops import stop_gradient
from mindspore.common.api import _pynative_executor
from ..configs import GENERATOR_SEED
from mindspore._c_expression import Cell_
from .grad_mode import no_grad

import mindtorch

grad_ = GradOperation(False, True, False)
grad_sens_ = GradOperation(False, True, True)
grad_input_sens_ = GradOperation(True, False, True)

def value_and_grad(fn, params_or_argnums, has_aux=False, attach_grads=True):
    use_argnums = False
    if isinstance(params_or_argnums, Generator):
        params_or_argnums = tuple(params_or_argnums)

    if isinstance(params_or_argnums[0], int):
        use_argnums = True

    def fn_aux(*args):
        outputs = fn(*args)
        no_grad_outputs = ()
        for out in outputs[1:]:
            no_grad_outputs += (stop_gradient(out),)
        return outputs[0], no_grad_outputs

    if has_aux:
        fn_ = fn_aux
    else:
        fn_ = fn

    def value_and_grad_f(*args, **kwargs):
        _pynative_executor.set_grad_flag(True)
        _pynative_executor.new_graph(fn, *args, **kwargs)
        values = fn_(*args, **kwargs)
        _pynative_executor.end_graph(fn, values, *args, **kwargs)

        run_args = args
        if kwargs:
            run_args = args + tuple(kwargs.values())

        if GENERATOR_SEED:
            grads = _pynative_executor.grad(fn_, grad_, params_or_argnums, None, *run_args)
            # grads = grad_(fn_, params)(*args, *params)
        else:
            _pynative_executor.grad(fn_, grad_, params_or_argnums, None, *run_args)
            grads = _pynative_executor() # pylint: disable=not-callable
        grads = tuple(mindspore.Tensor(grad) for grad in grads)
        if attach_grads:
            for param, grad in zip(params_or_argnums, grads):
                if param.grad is None:
                    param.grad = grad
                else:
                    param.grad += grad
            return values
        return values, grads
    return value_and_grad_f

def grad(fn, params_or_argnums=None, has_aux=False):
    value_and_grad_f = value_and_grad(fn, params_or_argnums, has_aux)
    def grad_f(*args):
        _, g = value_and_grad_f(*args)
        return g
    return grad_f

def vjp(fn, *inputs):
    grad_ = grad_input_sens_

    _pynative_executor.set_grad_flag(True)
    _pynative_executor.new_graph(fn, *inputs)
    values = fn(*inputs)
    _pynative_executor.end_graph(fn, values, *inputs)

    def wrap_container(*v):
        sens = v
        if len(v) == 1:
            sens = v[0]

        grads = _pynative_executor.grad(fn, grad_, None, None, *inputs, sens)
        return grads
    return values, wrap_container


class Function(Cell_):
    def __init__(self):
        super().__init__(str(self.__class__)[8:-2])
        self.saved_tensors = []
        self.used_bprop_inputs = []

    def save_for_backward(self, *args):
        if isinstance(args, tuple):
            self.saved_tensors.extend(list(args))
        else:
            self.saved_tensors.append(args)

    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *args, **kwargs):
        raise NotImplementedError

    def construct(self, *args, **kwargs):
        self.needs_input_grad = [input_.requires_grad if hasattr(input_, 'requires_grad') else False for input_ in args]
        args = (self,) + args
        outputs = self.forward(*args, **kwargs)
        self.device = outputs[0].device if isinstance(outputs, tuple) else outputs.device
        return outputs

    def bprop(self, *args, **kwargs):
        grads = args[-1]
        if isinstance(grads, tuple):
            grads = (mindtorch.Tensor(grad.stub, device=self.device) for grad in grads)
        else:
            grads = mindtorch.Tensor(grads.stub, device=self.device)
        args = (grads,)
        args = (self,) + args
        return self.backward(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with no_grad():
            output = self.construct(*args, **kwargs)
        _pynative_executor.call_custom_bprop(self, output, *args, **kwargs)
        return output

    @classmethod
    def apply(cls, *args, **kwargs):
        return cls()(*args, **kwargs)

def once_differentiable(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args):
        with mindtorch.no_grad():
            outputs = fn(ctx, *args)

        if not mindtorch.is_grad_enabled():
            return outputs

        # If any of the inputs have requires_grad=True, we force the outputs
        # to have requires_grad=True but point to a grad_fn which throws an
        # error message during (double) back-propagation.
        # XXX: this is only an approximation of requires_grad - there's no way
        # to figure out if fn didn't use ctx.saved_tensors and as a result
        # some Tensors might require grad, even if no args do.
        # Unfortunately, this leads to unexpected error messages ("no nodes
        # require computing gradients"), but I don't have a better idea.
        # These functions would raise an error in backward anyway.
        requires_grad = any(
            isinstance(arg, mindtorch.Tensor) and arg.requires_grad for arg in args
        )
        if not requires_grad:
            return outputs

        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        err_fn = RuntimeError(
            b"trying to differentiate twice a function that was marked "
            b"with @once_differentiable",
            len(outputs),
        )

        # Create aliases of each output that has requires_grad=True. We need
        # at least one of the inputs to err_fn to require grad so that the
        # output will have a grad_fn.
        def fake_requires_grad(var):
            if var is not None:
                var = var.detach()
                var.requires_grad = True
            return var

        return err_fn(*[fake_requires_grad(v) for v in outputs])