import mindtorch

def compute_requires_grad(*args):
    if not mindtorch.is_grad_enabled():
        return False
    
    requires_grad = False
    for arg in args:
        if isinstance(arg, mindtorch.Tensor):
            if arg.requires_grad:
                requires_grad = True
    return requires_grad
