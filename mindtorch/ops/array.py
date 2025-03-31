"""array op"""
import numbers
from mindspore.ops.auto_generate.gen_arg_handler import str_to_enum
from mindspore._c_expression import Tensor as MSTensor
import numpy as np

import mindtorch
from mindtorch.executor import execute
from .other import broadcast_tensors, broadcast_to

def t(input):
    assert input.ndim <= 2
    if input.ndim == 2:
        return transpose(input, 0, 1)
    return input

# adjoint

# argwhere
def argwhere(input):
    return execute('nonzero', input)

# cat
def cat(tensors, dim=0):
    device = tensors[0].device
    return execute('concat', tensors, dim, device=device)

# concat
def concat(tensors, dim=0):
    return cat(tensors, dim)

# concatenate
def concatenate(tensors, dim=0):
    return cat(tensors, dim)

# conj
def conj(input):
    return execute('conj', input)

# chunk
def chunk(input, chunks, dim=0):
    return execute('chunk', input, chunks, dim)

# dsplit


# column_stack


# dstack


# gather
def gather(input, dim, index):
    return execute('gather_d', input, dim, index)

def gather_nd(input, indices):
    return execute('gather_nd', input, indices)

# hsplit


# hstack

# index_fill


# index_add
def index_add(input, dim, index, source, *, alpha=1):
    return execute('index_add_ext', input, index, source, dim, alpha)

# index_copy


# index_reduce


# index_select
def index_select(input, dim, index):
    return execute('index_select', input, dim, index)

# masked_select
def masked_select(input, mask):
    return execute('masked_select', input, mask)

# movedim


# moveaxis


# narrow
def narrow(input, dim, start, length):
    return execute('narrow', input, dim, start, length)

# narrow_copy


# nonzero
def nonzero(input, *, as_tuple=False):
    if as_tuple:
        return execute('non_zero_ext', input)
    return execute('non_zero', input)

# permute
def permute(input, dims):
    assert isinstance(dims, tuple)
    return execute('transpose', input, dims)

# reshape
def reshape(input, *shape):
    if isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return execute('reshape', input, shape)

def view(input, *shape):
    return reshape(input, shape)

# row_stack

# select
def select(input, dim, index):
    return execute('select_ext', input, dim, index)

# scatter
def scatter(input, dim, index, src):
    if not isinstance(src, mindtorch.Tensor):
        return execute('scatter_value', input, dim, index, src, str_to_enum('ScatterValue', 'reduce', 'none'))
    return execute('scatter', input, dim, index, src, str_to_enum('Scatter', 'reduce', 'none'))

# diagonal_scatter


# select_scatter


# slice_scatter


# scatter_add
def scatter_add(input, dim, index, src):
    return execute('scatter_add_ext', input, dim, index, src)

# scatter_reduce


# split
def split(tensor, split_size_or_sections, dim=0):
    if isinstance(split_size_or_sections, int):
        res = execute('split_tensor', tensor, split_size_or_sections, dim)
    elif isinstance(split_size_or_sections, (list, tuple)):
        res = execute('split_with_size', tensor, split_size_or_sections, dim)
    else:
        raise TypeError(f"Type of Argument `split_size_or_sections` should be integer, tuple(int) or list(int), "
                        f"but got {type(split_size_or_sections)}")
    return res

# squeeze
def squeeze(input, dim=None):
    return execute('squeeze', input, dim)

# stack

def stack(tensors, dim=0):
    device = tensors[0].device
    if device.type == 'cpu':
        return execute('stack', tensors, dim, device=device)
    return execute('stack_ext', tensors, dim, device=device)

# swapaxes
def swapaxes(input, dim0, dim1):
    return transpose(input, dim0, dim1)

# swapdims
def swapdims(input, dim0, dim1):
    return transpose(input, dim0, dim1)

# take
def take(input, index):
    input = input.view(-1)
    index_shape = index.shape
    index = index.view(-1)
    return gather(input, 0, index).view(index_shape)

# take_along_dim


# tensor_split


# tile
def tile(input, dims):
    return execute('tile', input, dims)

# transpose
def transpose(input, dim0, dim1):
    return execute('transpose_ext', input, dim0, dim1)

# unbind
def unbind(input, dim=0):
    return execute('unstack_ext', input, dim)

# unravel_index

# unsqueeze
def unsqueeze(input, dim):
    return execute('expand_dims', input, dim)

# vsplit

# vstack

# where
def where(condition, *args):
    if len(args) == 0:
        return nonzero(condition, as_tuple=True)
    assert len(args) == 2
    input, other = args
    return execute('select', condition, input, other)



tensor_1d = MSTensor([0], dtype=mindtorch.int64)
empty_tensor_1d = MSTensor(shape=(0,), dtype=mindtorch.int64)
empty_tensor_9d = MSTensor(shape=(0,)*9, dtype=mindtorch.int64)

def _do_select(self, dim: int, index: int, dim_index: int, self_shape: list):
    """call select view operator"""
    if not self_shape:
        raise TypeError("Invalid index of a 0-dim tensor.")
    dim_size = self_shape[dim]
    if index >= dim_size or index < -dim_size:
        raise IndexError(f"Index {index} is out of bounds for dimension {dim_index} with size {dim_size}")
    index = index + dim_size if index < 0 else index
    return execute('select_ext', self, dim, index)


def _do_slice(self, dim: int, index: slice, self_shape: list):
    """call slice view operator"""
    def _get_index(index, default):
        if index is None:
            return default
        if mindtorch.is_tensor(index):
            index = int(index)
        return index

    if not self_shape:
        raise TypeError("Invalid index of a 0-dim tensor.")
    step = _get_index(index.step, 1)
    if step <= 0:
        raise ValueError("slice step must be positive")
    start = _get_index(index.start, 0)
    end = _get_index(index.stop, self_shape[dim])
    if start == 0 and end == self_shape[dim] and step == 1:
        return self
    return execute('slice_ext', self, dim, start, end, step)

def _wrap_index_to_tuple(index):
    """Wrap index to tuple"""
    if isinstance(index, tuple):
        return index
    if isinstance(index, list):
        if len(index) < 32 and any(isinstance(i, (mindtorch.Tensor, list, tuple, slice, type(None), type(...))) for i in index):
            return tuple(index)
    return (index,)


def _count_indexed_dims(indexes):
    """Count indexed dims"""
    count = 0
    for index in indexes:
        if isinstance(index, mindtorch.Tensor):
            if index.dtype == mindtorch.bool:
                count += index.ndim
            else:
                count += 1
        elif not isinstance(index, (type(None), type(...), bool)):
            count += 1
    return count

def _record_tensor_index(index, remain_indexes, dim):
    """Record indexes remained to be used by aclnnIndex/aclnnIndexPut"""
    if len(remain_indexes) > dim:
        remain_indexes[dim] = index
        return remain_indexes

    while dim > len(remain_indexes):
        # use empty_tensor with dim_num 9 to indicate unused dim
        remain_indexes.append(empty_tensor_9d)

    remain_indexes.append(index)
    return remain_indexes

def _process_dim_in_multi_dim_index(prev_result, orig_tensor, index, dim, indexed_dims, dim_index, remain_indexes,
                                    prev_shape):
    """Process dim in multi dim index"""
    if isinstance(index, bool):
        result = unsqueeze(prev_result, dim)
        index_for_bool = tensor_1d if index else empty_tensor_1d
        _record_tensor_index(index_for_bool, remain_indexes, dim)
        prev_shape.insert(dim, 1)
        dim += 1
        return result, dim, remain_indexes, prev_shape
    if isinstance(index, int):
        result = _do_select(prev_result, dim, index, dim_index, prev_shape)
        del prev_shape[dim]
        return result, dim, remain_indexes, prev_shape
    if isinstance(index, slice):
        result = _do_slice(prev_result, dim, index, prev_shape)
        # current dim in prev_shape will not be used later, ignore it
        dim += 1
        return result, dim, remain_indexes, prev_shape
    if isinstance(index, type(...)):
        dim += (orig_tensor.ndim - indexed_dims)
        return prev_result, dim, remain_indexes, prev_shape
    if index is None:
        result = unsqueeze(prev_result, dim)
        prev_shape.insert(dim, 1)
        dim += 1
        return result, dim, remain_indexes, prev_shape
    if isinstance(index, mindtorch.Tensor):
        result = prev_result
        if index.ndim == 0 and index.dtype in (mindtorch.int, mindtorch.long, mindtorch.short, mindtorch.bool):
            if index.dtype in (mindtorch.int, mindtorch.long, mindtorch.short):
                result = _do_select(prev_result, dim, index.item(), dim_index, prev_shape)
                del prev_shape[dim]
                return result, dim, remain_indexes, prev_shape
            # process index with Tensor bool type
            result = unsqueeze(prev_result, dim)
            index_for_bool = tensor_1d if index else empty_tensor_1d
            _record_tensor_index(index_for_bool, remain_indexes, dim)
            prev_shape.insert(dim, 1)
            dim += 1
            return result, dim, remain_indexes, prev_shape
        _record_tensor_index(index, remain_indexes, dim)
        dim += 1
        return result, dim, remain_indexes, prev_shape
    raise IndexError(f"Invalid tensor index type {index}")


def _process_multi_dim_index(self, indexes, remain_indexes, indexed_dims):
    """Process indexes in tuple"""
    self_viewed = self
    self_viewed_shape = list(self.shape)
    dim = 0
    for i, index in enumerate(indexes):
        if isinstance(index, (list, tuple, np.ndarray)):
            index_np = np.array(index) if isinstance(index, (list, tuple)) else index
            if index_np.dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,
                                  np.float16, np.float32, np.float64):
                index = mindtorch.tensor(index_np, device=self.device, dtype=mindtorch.int64)
            elif index_np.dtype == np.bool_:
                index = mindtorch.tensor(index_np, device=self.device, dtype=mindtorch.int64)
            else:
                raise TypeError(f"Index {index} contain unsupported elements")
        self_viewed, dim, remain_indexes, self_viewed_shape = _process_dim_in_multi_dim_index(
            self_viewed, self, index, dim, indexed_dims, i, remain_indexes, self_viewed_shape)
    return self_viewed, remain_indexes


def tensor_getitem(self, index):
    """Handle tensor getitem"""
    if isinstance(index, bool):
        self_viewed = unsqueeze(self, 0)
        index_for_bool = tensor_1d if index else empty_tensor_1d
        return execute('index', self_viewed, [index_for_bool])
    if isinstance(index, int):
        return _do_select(self, 0, index, 0, list(self.shape))
    if isinstance(index, slice):
        result = _do_slice(self, 0, index, list(self.shape))
        return result
    if index is None:
        return unsqueeze(self, 0)
    if isinstance(index, type(...)):
        return self
    indexes = _wrap_index_to_tuple(index)
    indexed_dims = _count_indexed_dims(indexes)
    if self.ndim < indexed_dims:
        raise IndexError(f"too many indices for tensor with dimension size {self.ndim}")
    remain_indexes = []
    self_viewed, remain_indexes = _process_multi_dim_index(self, indexes, remain_indexes, indexed_dims)
    if not remain_indexes:
        return self_viewed
    return execute('index', self_viewed, remain_indexes)


def tensor_setitem(self, index, value):
    """Handle tensor setitem"""
    if not isinstance(value, mindtorch.Tensor):
        if isinstance(value, (bool, int, float)):
            value = mindtorch.tensor(value, dtype=self.dtype, device=self.device)
        else:
            raise TypeError(f"Can't assign a {type(value)} to a {self.dtype}.")

    if isinstance(index, bool) and index is False:
        return self
    if isinstance(index, type(...)):
        execute('inplace_copy', self, value)
        return self
    if index is None or (isinstance(index, bool) and index is True):
        self_viewed = unsqueeze(self, 0)
        execute('inplace_copy', self_viewed, value)
        return self
    if isinstance(index, int):
        self_viewed = _do_select(self, 0, index, 0, list(self.shape))
        execute('inplace_copy', self_viewed, value)
        return self
    if isinstance(index, slice):
        self_viewed = _do_slice(self, 0, index, list(self.shape))
        execute('inplace_copy', self_viewed, value)
        return self
    indexes = _wrap_index_to_tuple(index)
    indexed_dims = _count_indexed_dims(indexes)
    if self.ndim < indexed_dims:
        raise IndexError(f"too many indices for tensor with dimension size {self.ndim}")
    remain_indexes = []
    self_viewed, remain_indexes = _process_multi_dim_index(self, indexes, remain_indexes, indexed_dims)
    if not remain_indexes:
        execute('inplace_copy', self_viewed, value)
        return self
    execute('inplace_index_put', self_viewed, remain_indexes, value, False) # accumulate=False
    return self

_SLICE_ERROR = (
    'only integers, slices (`:`), ellipsis (`...`), '
    'newaxis (`None`) and integer or boolean arrays are valid indices'
)

def _as_index(idx, need_scalar=True):
    """Helper function to parse idx as an index.
    """
    if isinstance(idx, numbers.Integral):
        return idx, True

    idx = mindtorch.tensor(idx)
    if need_scalar and idx.ndim not in (None, 0):
        raise IndexError(_SLICE_ERROR + ', got {!r}'.format(idx))

    if idx.ndim == 0:
        return idx.item(), True
    return idx, False

def cumprod(x, axis=0, exclusive=False, reverse=False):
    x = np.array(x)
    if reverse:
        x = np.flip(x, axis=axis)

    if exclusive:
        shifted_x = np.ones_like(x)
        if axis == 0:
            shifted_x[1:] = x[:-1]
        else:
            shifted_x[:, 1:] = x[:, :-1]
        result = np.cumprod(shifted_x, axis=axis)
    else:
        result = np.cumprod(x, axis=axis)

    if reverse:
        result = np.flip(result, axis=axis)

    return result

def moveaxis(a, source, destination):
    """Raises ValueError if source, destination not in (-ndim(a), ndim(a))."""
    if not source and not destination:
        return a

    if isinstance(source, int):
        source = (source,)
    if isinstance(destination, int):
        destination = (destination,)
    if len(source) != len(destination):
        raise ValueError('The lengths of source and destination must equal')

    a_rank = a.ndim

    def _correct_axis(axis, rank):
        if axis < 0:
            return axis + rank
        return axis

    source = tuple(_correct_axis(axis, a_rank) for axis in source)
    destination = tuple(_correct_axis(axis, a_rank) for axis in destination)

    if a.ndim is not None:
        perm = [i for i in range(a_rank) if i not in source]
        for dest, src in sorted(zip(destination, source)):
            assert dest <= len(perm)
            perm.insert(dest, src)
    else:
        r = mindtorch.range(0, a_rank, 1)

        def _remove_indices(a, b):
            """Remove indices (`b`) from `a`."""
            items = mindtorch.unbind(
                mindtorch.sort(mindtorch.stack(b))
            )

            i = 0
            result = []

            for item in items:
                result.append(a[i:item])
                i = item + 1

            result.append(a[i:])

            return mindtorch.concat(result, 0)

        minus_sources = _remove_indices(r, source)
        minus_dest = _remove_indices(r, destination)

        perm = execute('scatter_nd', 
            mindtorch.unsqueeze(minus_dest, 1), minus_sources, [a_rank]
        )
        perm = execute('tensor_scatter_update',
            perm, mindtorch.unsqueeze(destination, 1), source
        )
    a = mindtorch.permute(a, tuple(perm))

    return a

def _slice_helper(tensor, slice_spec, do_update=False, updates=None):
    """Helper function for __getitem__ and _with_index_update_helper.
    """
    begin, end, strides = [], [], []
    new_axis_mask, shrink_axis_mask = 0, 0
    begin_mask, end_mask = 0, 0
    ellipsis_mask = 0
    advanced_indices = []
    shrink_indices = []
    for index, s in enumerate(slice_spec):
        if isinstance(s, slice):
            if s.start is not None:
                begin.append(s.start)
            else:
                begin.append(0)
                begin_mask |= (1 << index)
            if s.stop is not None:
                end.append(s.stop)
            else:
                end.append(0)
                end_mask |= (1 << index)
            if s.step is not None:
                strides.append(s.step)
            else:
                strides.append(1)
        elif s is Ellipsis:
            begin.append(0)
            end.append(0)
            strides.append(1)
            ellipsis_mask |= (1 << index)
        elif s is None:
            # begin.append(0)
            # end.append(0)
            # strides.append(1)
            new_axis_mask |= (1 << index)
        else:
            s, is_scalar = _as_index(s, False)
            if is_scalar:
                begin.append(s)
                end.append(s + 1)
                strides.append(1)
                shrink_axis_mask |= (1 << index)
                shrink_indices.append(index)
            else:
                begin.append(0)
                end.append(0)
                strides.append(1)
                begin_mask |= (1 << index)
                end_mask |= (1 << index)
                advanced_indices.append((index, s, ellipsis_mask != 0))

    if do_update and not advanced_indices:
        return strided_slice_update(
            tensor,
            begin,
            end,
            strides,
            updates,
            begin_mask=begin_mask,
            end_mask=end_mask,
            shrink_axis_mask=shrink_axis_mask,
            new_axis_mask=new_axis_mask,
            ellipsis_mask=ellipsis_mask,
        )
    else:
        if updates is not None:
            original_tensor = tensor
        tensor = execute(
            'strided_slice',
            tensor,
            begin,
            end,
            strides,
            begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask
        )

    if not advanced_indices:
        return tensor

    advanced_indices_map = {}
    for index, data, had_ellipsis in advanced_indices:
        if had_ellipsis:
            num_shrink = len([x for x in shrink_indices if x > index])
            dim = index - len(slice_spec) + num_shrink
        else:
            num_shrink = len([x for x in shrink_indices if x < index])
            dim = index - num_shrink
        advanced_indices_map[dim] = data
    dims = sorted(advanced_indices_map.keys())
    dims_contiguous = True
    if len(dims) > 1:
        if dims[0] < 0 and dims[-1] >= 0:  # not all same sign
            dims_contiguous = False
        else:
            for i in range(len(dims) - 1):
                if dims[i] + 1 != dims[i + 1]:
                    dims_contiguous = False
                    break
    indices = [advanced_indices_map[x] for x in dims]
    indices = broadcast_tensors(*indices)
    stacked_indices = stack(indices, dim=-1)
    # Skip the contiguous-dims optimization for update because there is no
    # tf.*scatter* op that supports the `axis` argument.
    if not dims_contiguous or updates is not None:
        if range(len(dims)) != dims:
            tensor = moveaxis(tensor, dims, range(len(dims)))
        tensor_shape_prefix = mindtorch.tensor(tensor.shape[: len(dims)])
        stacked_indices = where(
            stacked_indices < 0,
            stacked_indices + tensor_shape_prefix,
            stacked_indices,
        )
        if updates is None:
            return execute('gather_nd', tensor, stacked_indices)
        else:
            # We only need to move-axis `updates` in the contiguous case becausce
            # only in this case the result dimensions of advanced indexing are in
            # the middle of `updates`. In the non-contiguous case, those dimensions
            # are always at the front.
            if dims_contiguous:
                batch_size = stacked_indices.ndim - 1
                batch_start = dims[0]
                if batch_start < 0:
                    batch_start += len(dims) - batch_size

                def range_(start, length):
                    return range(start, start + length)

                updates = moveaxis(
                    updates, range_(batch_start, batch_size), range(batch_size)
                )
            tensor = execute('tensor_scatter_update', tensor, stacked_indices, updates)
            if range(len(dims)) != dims:
                tensor = moveaxis(tensor, range(len(dims)), dims)
            return strided_slice_update(
                original_tensor,
                begin,
                end,
                strides,
                tensor,
                begin_mask=begin_mask,
                end_mask=end_mask,
                shrink_axis_mask=shrink_axis_mask,
                new_axis_mask=new_axis_mask,
                ellipsis_mask=ellipsis_mask,
            )

    # Note that gather_nd does not support gathering from inside the array.
    # To avoid shuffling data back and forth, we transform the indices and
    # do a gather instead.
    rank = tensor.ndim
    dims = [(x + rank if x < 0 else x) for x in dims]
    shape_tensor = tensor.shape
    dim_sizes = np.take_along_axis(np.array(shape_tensor), np.array(dims), axis=0)
    if len(dims) == 1:
        stacked_indices = indices[0]
    stacked_indices = stacked_indices.to(mindtorch.int32)
    stacked_indices = where(
        stacked_indices < 0, stacked_indices + mindtorch.tensor(dim_sizes), stacked_indices
    )
    axis = dims[0]
    if len(dims) > 1:
        index_scaling = cumprod(dim_sizes, reverse=True, exclusive=True)

        def _tensordot(a, b):
            # TODO(b/168657656): This function should be replaced by
            # tensordot(axis=1) once MatMul has int32 XLA kernel.
            b = broadcast_to(b, a.shape)
            return mindtorch.sum(a * b, dim=-1)

        stacked_indices = _tensordot(stacked_indices, mindtorch.tensor(index_scaling))
        flat_shape = shape_tensor[:axis] + (-1,) + shape_tensor[axis + len(dims) :]
        tensor = tensor.reshape(flat_shape)

    return gather(tensor, axis, stacked_indices)

def _as_spec_tuple(slice_spec):
    """Convert slice_spec to tuple."""
    if isinstance(slice_spec, (list, tuple)):
        is_index = True
        for s in slice_spec:
            if s is None or s is Ellipsis or isinstance(s, (list, tuple, slice)):
                is_index = False
                break
        if not is_index:
            return tuple(slice_spec)
    return (slice_spec,)

def getitem(self, slice_spec):
    if (
        isinstance(slice_spec, bool)
        or (
            isinstance(slice_spec, mindtorch.Tensor)
            and slice_spec.dtype == mindtorch.bool
        )
    ):
        return masked_select(self, slice_spec)

    if not isinstance(slice_spec, tuple):
        slice_spec = _as_spec_tuple(slice_spec)

    result_t = _slice_helper(self, slice_spec)
    return result_t

def setitem(a, slice_spec, updates):
    """Implementation of ndarray._with_index_*."""
    if (
        isinstance(slice_spec, bool)
        or (
            isinstance(slice_spec, mindtorch.Tensor)
            and slice_spec.dtype == mindtorch.bool
        )
    ):
        slice_spec = nonzero(slice_spec)

    if not isinstance(slice_spec, tuple):
        slice_spec = _as_spec_tuple(slice_spec)

    a_dtype = a.dtype
    result_t = _slice_helper(a, slice_spec, True, updates)
    return result_t.to(a_dtype)

def strided_slice_update(input, begin, end, strides, update, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
    if isinstance(update, (int, float, bool)):
        update = mindtorch.tensor(update, device=input.device, dtype=input.dtype)
    sliced_tensor = execute('strided_slice', input, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)
    if update.shape != sliced_tensor.shape:
        update = update.broadcast_to(sliced_tensor.shape)
    updated_tensor = execute('strided_slice_grad', input, begin, end, strides, update, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)
    out = where(updated_tensor != input, updated_tensor, input)
    input.data = out
    return input

__all__ = [
    # adjoint,
    'argwhere',
    'cat',
    'concat',
    'concatenate',
    'conj',
    'chunk',
    # dsplit,
    # column_stack
    # dstack
    'gather',
    'gather_nd',
    # hsplit
    'index_add',
    # index_copy
    # index_reduce
    'index_select',
    'masked_select',
    # movedim
    # moveaxis
    'narrow',
    # narrow_copy
    'nonzero',
    'permute',
    'reshape',
    'view',
    # row_stack
    'select',
    'scatter',
    # diagonal_scatter
    # select_scatter
    # slice_scatter
    'scatter_add',
    # scatter_reduce
    'split',
    'squeeze',
    'stack',
    'swapaxes',
    'swapdims',
    'take',
    # take_along_dim
    # tensor_split
    'tile',
    'transpose',
    'unbind',
    # unravel_index
    'unsqueeze',
    # vsplit
    'where',
    'tensor_getitem',
    'tensor_setitem',
    't',
    'getitem',
    'setitem'
]
