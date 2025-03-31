import sys
import traceback

import mindtorch

element_size_map = {
    mindtorch.float16: 2,
    mindtorch.float32: 3,
    mindtorch.bfloat16: 2,
    mindtorch.int64: 4,
    mindtorch.uint8: 1,
    mindtorch.int8: 1,
    mindtorch.bool_: 1
}

def _element_size(dtype):
    return element_size_map[dtype]

def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.

    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.

    Args:
        tensors (Iterable[Tensor]): dense tensors to flatten.

    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    tensors = [tensor.view(-1) for tensor in tensors]
    return mindtorch.cat(tensors)


def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.

    Args:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        if numel == 0:
            outputs.append(mindtorch.empty(0, flat.dtype))
        else:
            outputs.append(mindtorch.narrow(flat, 0, offset, numel).view(tensor.shape))
            offset += numel
    return outputs

def _rebuild_tensor_v2(
    storage,
    storage_offset,
    size,
    stride,
    requires_grad,
    backward_hooks,
    metadata=None,
):
    return mindtorch.Tensor(storage)

class KeyErrorMessage(str):
    r"""str subclass that returns itself in repr"""

    def __repr__(self):
        return self

class ExceptionWrapper:
    r"""Wraps an exception plus traceback to communicate across threads"""

    def __init__(self, exc_info=None, where="in background"):
        # It is important that we don't store exc_info, see
        # NOTE [ Python Traceback Reference Cycle Problem ]
        if exc_info is None:
            exc_info = sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))
        self.where = where

    def reraise(self):
        r"""Reraises the wrapped exception in the current thread"""
        # Format a message such as: "Caught ValueError in DataLoader worker
        # process 2. Original Traceback:", followed by the traceback.
        msg = f"Caught {self.exc_type.__name__} {self.where}.\nOriginal {self.exc_msg}"
        if self.exc_type == KeyError:
            # KeyError calls repr() on its argument (usually a dict key). This
            # makes stack traces unreadable. It will not be changed in Python
            # (https://bugs.python.org/issue2651), so we work around it.
            msg = KeyErrorMessage(msg)
        elif getattr(self.exc_type, "message", None):
            # Some exceptions have first argument as non-str but explicitly
            # have message field
            raise self.exc_type(message=msg)
        try:
            exception = self.exc_type(msg)
        except Exception:
            # If the exception takes multiple arguments or otherwise can't
            # be constructed, don't try to instantiate since we don't know how to
            raise RuntimeError(msg) from None
        raise exception
