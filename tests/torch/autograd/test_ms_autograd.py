import mindspore
import torch

def test_ms_autograd():
    x = torch.Tensor(3, 3)
    x.requires_grad = True
    def fn(y):
        return x * y
    
    grad_fn = mindspore.value_and_grad(fn, None, weights=(x,))
    out, grads = grad_fn(2)
    print(out, grads)
    print(x.grad)

def test_ms_autograd_origin():
    x = mindspore.Parameter(mindspore.ops.ones((3, 3)))
    y = mindspore.Tensor(2)
    def fn(y):
        return x * y
    
    grad_fn = mindspore.value_and_grad(fn, None, weights=(x,))
    out, grads = grad_fn(y)
    print(out, grads)
