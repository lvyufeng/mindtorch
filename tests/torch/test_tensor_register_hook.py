import torch

def test_middle_register_hook():
    with torch.autograd.GradientTape() as tape:
        x = torch.tensor(1., requires_grad=True, device=torch.device('npu'))
        assert x.is_leaf
        assert x.requires_grad
        assert x.attach_grad_hook is not None
        y = x + 1
        y.retain_grad()
        assert not y.is_leaf
        assert y.requires_grad
        assert y.attach_grad_hook is not None
        z = y * 2

    tape.gradient(z, (x,))
    print(x.grad)
    print(y.grad)