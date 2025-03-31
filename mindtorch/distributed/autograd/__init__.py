# mypy: allow-untyped-defs

import mindtorch


def is_available():
    return hasattr(mindtorch._C, "_dist_autograd_init")


if is_available() and not mindtorch._C._dist_autograd_init():
    raise RuntimeError("Failed to initialize mindtorch.distributed.autograd")

if is_available():
    from mindtorch._C._distributed_autograd import (
        _current_context,
        _get_debug_info,
        _get_max_id,
        _init,
        _is_valid_context,
        _new_context,
        _release_context,
        _retrieve_context,
        backward,
        DistAutogradContext,
        get_gradients,
    )


class context:
    """
    Context object to wrap forward and backward passes when using
    distributed autograd. The ``context_id`` generated in the ``with``
    statement  is required to uniquely identify a distributed backward pass
    on all workers. Each worker stores metadata associated with this
    ``context_id``, which is required to correctly execute a distributed
    autograd pass.

    Example::
        >>> # xdoctest: +SKIP
        >>> import mindtorch.distributed.autograd as dist_autograd
        >>> with dist_autograd.context() as context_id:
        >>>     t1 = mindtorch.rand((3, 3), requires_grad=True)
        >>>     t2 = mindtorch.rand((3, 3), requires_grad=True)
        >>>     loss = rpc.rpc_sync("worker1", mindtorch.add, args=(t1, t2)).sum()
        >>>     dist_autograd.backward(context_id, [loss])
    """

    def __enter__(self):
        self.autograd_context = _new_context()
        return self.autograd_context._context_id()

    def __exit__(self, type, value, traceback):
        _release_context(self.autograd_context._context_id())
