import torch
from torch.autograd import Function
import mindspore

class Test(Function):

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x + y + 1
    
    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.saved_tensors
        print(x, y)
        return torch.ones_like(x), torch.zeros_like(y)

def fn_test(x, y):
    return Test.apply(x, y)

def test_function_no_record_forward_inputs():
    x = torch.randn(3, 3)
    y = torch.randn(3)
    print(x, y)
    grad_fn = mindspore.value_and_grad(fn_test, (0, 1))
    value, grads = grad_fn(x, y)
    print(value, grads)
