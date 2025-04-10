import mindtorch

def is_tensor_like(inp):
    """
    Returns ``True`` if the passed-in input is a Tensor-like.

    Currently, this occurs whenever there's a ``__torch_function__``
    attribute on the type of the input.

    Examples
    --------
    A subclass of tensor is generally a Tensor-like.

    >>> class SubTensor(torch.Tensor): ...
    >>> is_tensor_like(SubTensor([0]))
    True

    Built-in or user types aren't usually Tensor-like.

    >>> is_tensor_like(6)
    False
    >>> is_tensor_like(None)
    False
    >>> class NotATensor: ...
    >>> is_tensor_like(NotATensor())
    False

    But, they can be made Tensor-like by implementing __torch_function__.

    >>> class TensorLike:
    ...     @classmethod
    ...     def __torch_function__(cls, func, types, args, kwargs):
    ...         return -1
    >>> is_tensor_like(TensorLike())
    True
    """
    return type(inp) is mindtorch.Tensor or hasattr(inp, "__torch_function__")
