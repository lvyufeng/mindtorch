import sys
import torch

def test_torch_autograd():
    x = torch.tensor(1.)
    x.requires_grad = True
    y = x + 1
    print(y)
    print(y.device)
    y.backward()
    print(x.grad)

def test_torch_autograd_retain_grad():
    x = torch.tensor(1.)
    print(sys.getrefcount(x))
    y = torch.tensor(2.)
    x.requires_grad = True
    z = x + y
    zz = z + 1
    print(z.grad)
    z.retain_grad()
    zz.backward()
    print(sys.getrefcount(x))

    # z = x + y
    # zz = z + 1
    # print(z.grad)
    # z.retain_grad()
    # zz.backward()
    print(z.grad)
    print(x.grad)
    print(y.grad)
    print(sys.getrefcount(x))
