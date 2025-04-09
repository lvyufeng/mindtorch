"""Parameter"""
import mindtorch
from typing import OrderedDict

class Parameter(mindtorch.Tensor):
    r"""A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~mindtorch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Args:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. Note that
            the mindtorch.no_grad() context does NOT affect the default behavior of
            Parameter creation--the Parameter will still have `requires_grad=True` in
            :class:`~no_grad` mode. See :ref:`locally-disable-grad-doc` for more
            details. Default: `True`
    """

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = mindtorch.empty(0)
        if type(data) is mindtorch.Tensor or type(data) is Parameter:
            # For ease of BC maintenance, keep this path for standard Tensor.
            # Eventually (tm), we should change the behavior for standard Tensor to match.
            return mindtorch.Tensor._make_subclass(cls, data, requires_grad)

        # Path for custom tensors: set a flag on the instance to indicate parameter-ness.
        t = data.detach().requires_grad_(requires_grad)
        if type(t) is not type(data):
            raise RuntimeError(
                f"Creating a Parameter from an instance of type {type(data).__name__} "
                "requires that detach() returns an instance of the same type, but return "
                f"type {type(t).__name__} was found instead. To use the type as a "
                "Parameter, please correct the detach() semantics defined by "
                "its __torch_dispatch__() implementation."
            )
        t._is_param = True
        return t

    # Note: the 3 methods below only apply to standard Tensor. Parameters of custom tensor types
    # are still considered that custom tensor type and these methods will not be called for them.
    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(
                self.data.clone(memory_format=mindtorch.preserve_format), self.requires_grad
            )
            memo[id(self)] = result
            return result

    def __repr__(self):
        return "Parameter containing:\n" + super().__repr__()

    # def __reduce_ex__(self, proto):
    #     state = mindtorch._utils._get_obj_state(self)

    #     # See Note [Don't serialize hooks]
    #     hooks = OrderedDict()
    #     if not state:
    #         return (
    #             mindtorch._utils._rebuild_parameter,
    #             (self.data, self.requires_grad, hooks),
    #         )

    #     return (
    #         mindtorch._utils._rebuild_parameter_with_state,
    #         (self.data, self.requires_grad, hooks, state),
    #     )
