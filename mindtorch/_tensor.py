import uuid
import weakref
import warnings
from copy import deepcopy
import numpy as np

import mindspore
from mindspore._c_expression import Tensor as MSTensor, ParamInfo
from mindspore._c_expression import TensorNode
from mindspore.ops import GradOperation
from mindspore.common.api import _pynative_executor
from mindspore._c_expression import typing

import mindtorch
from mindtorch.dispatcher import device_map
from .types import device as device_
from ._utils import _rebuild_tensor_v2
from ._C.size import Size

grad_ = GradOperation(False, True, True)

kMaxInt8 = 2 ** 7 - 1
kMaxInt16 = 2 ** 15 - 1
kMaxInt32 = 2 ** 31 - 1
kMaxInt64 = 2 ** 63 - 1
kMaxUint8 = 2 ** 8 - 1
kMaxUint16 = 2 ** 16 - 1
kMaxUint32 = 2 ** 32 - 1
kMaxUint64 = 2 ** 64 - 1
kMantissaFloat16 = 2 ** 11
kMantissaFloat32 = 2 ** 24
kMantissaFloat64 = 2 ** 53


class TensorMeta(type):
    def __instancecheck__(self, instance):
        if self == Tensor and type(instance) == mindtorch.nn.Parameter:
            return True
        dtype = getattr(instance, 'dtype', None)
        if dtype is not None and dtype in dtype_class_map:
            return self == dtype_class_map[instance.dtype]
        return super().__instancecheck__(instance)

class Tensor(metaclass=TensorMeta):
    tensor = None
    stub = None
    grad = None

    _base = None # notice: _base should be root Tensor when created by view class op
    _user_created = False
    _requires_grad = False
    _retain_grad = False

    def __init__(self, *input, device=None, dtype=None): # pylint: disable=super-init-not-called
        if hasattr(self, '_is_param') and self._is_param:
            return
        if device is None:
            device = device_('cpu')
        self.device = device

        if isinstance(input[0], TensorNode):
            self.stub = input[0]
        elif isinstance(input[0], MSTensor):
            self.tensor = input[0]
        elif isinstance(input[0], int):
            if dtype is None:
                dtype = mindspore.float32
            self.tensor = MSTensor(shape=input, dtype=dtype)
            self._user_created = True
        elif isinstance(input[0], np.ndarray):
            self.tensor = MSTensor(input[0])
            self._user_created = True
        elif isinstance(input[0], Tensor):
            self.tensor = input[0].tensor
            self.stub = input[0].stub
        else:
            raise ValueError(f'not support data type {type(input[0])}')

    @staticmethod
    def _make_subclass(cls, tensor, requires_grad=None):
        """
        Manually implement the functionality of mindtorch.Tensor._make_subclass.

        Args:
            cls (type): The subclass type to create.
            tensor (mindtorch.Tensor): The base tensor to wrap.
            requires_grad (bool, optional): Whether the subclassed tensor requires gradients.

        Returns:
            cls: An instance of the subclass wrapping the base tensor.
        """
        if not issubclass(cls, mindtorch.Tensor):
            raise TypeError(f"{cls} must be a subclass of mindtorch.Tensor")
        
        # Create an uninitialized instance of the subclass
        subclass_instance = object.__new__(cls)
        
        # Initialize the core structure of the tensor
        Tensor.__init__(subclass_instance, tensor, device=tensor.device)


        if cls == mindtorch.nn.Parameter:
            subclass_instance._is_param = True
            subclass_instance._user_created = True

        # Set requires_grad if specified; otherwise, inherit from the base tensor
        if requires_grad is not None:
            subclass_instance.requires_grad_(requires_grad)
        else:
            subclass_instance.requires_grad_(tensor.requires_grad)

        return subclass_instance

    @property
    def _data(self):
        return self.stub_sync()

    @property
    def data(self):
        return Tensor(MSTensor(self._data), device=self.device)

    @data.setter
    def data(self, other):
        if isinstance(other, Tensor):
            if self.stub is not None and other.stub is not None:
                self.stub = other.stub
            else:
                self._data.assign_value_cpp(other._data)
            self.device = other.device
        else:
            raise ValueError(f'not support set type {type(other)} to Tensor.data')

    @property
    def requires_grad(self):
        if not _pynative_executor.requires_grad():
            warnings.warn('`requires_grad` does not take effect when it is not within the scope of the differential function.')

        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad):
        if not isinstance(self.dtype, (typing.Float, typing.BFloat)) and requires_grad:
            raise RuntimeError('only Tensors of floating point and complex dtype can require gradients')
        self._requires_grad = requires_grad
        if requires_grad:
            if self.is_leaf:
                if self._data.param_info is None:
                    self._data.param_info = ParamInfo()
                    self._data.param_info.name = str(uuid.uuid4())
                self._data.param_info.requires_grad = requires_grad
                if not hasattr(self, 'attach_grad_hook'):
                    self.attach_grad()
                    self._retain_grad = True
        else:
            if self.is_leaf:
                if self._data.param_info is not None:
                    self._data.param_info = None
            if hasattr(self, 'attach_grad_hook'):
                # TODO: remove handle
                pass

    def attach_grad(self):
        weak_self = weakref.ref(self)
        def hook(grad):
            param = weak_self()
            if param is not None and param._retain_grad:
                if param.grad is None:
                    param.grad = Tensor((grad * 1).stub, device=param.device)
                else:
                    param.grad += Tensor((grad * 1).stub, device=param.device)
            return grad
        self.attach_grad_hook = self.register_hook(hook)

    def retain_grad(self):
        if not self.requires_grad:
            raise RuntimeError("can't retain_grad on Tensor that has requires_grad=False")
        self._retain_grad = True

    @property
    def shape(self):
        """shape stub."""
        if self.stub:
            if not hasattr(self, "stub_shape"):
                self.stub_shape = self.stub.get_shape()
            return Size(self.stub_shape)
        return Size(self.tensor.shape)

    @property
    def dtype(self):
        """dtype stub."""
        if self.stub:
            if not hasattr(self, "stub_dtype"):
                self.stub_dtype = self.stub.get_dtype()
            return self.stub_dtype
        return self.tensor.dtype

    def cpu(self):
        return self.to(device_('cpu'))

    def npu(self, device=None, non_blocking=False):
        if device is None:
            device = device_('npu', 0)
        return self.to(device, non_blocking=non_blocking)

    def cuda(self, device=None, non_blocking=False):
        if device is None:
            device = device_('gpu', 0)
        return self.to(device, non_blocking=non_blocking)

    def requires_grad_(self, requires_grad=True):
        self.requires_grad = requires_grad

    def __reduce_ex__(self, proto):
        storage_offset = 0
        size = self.shape
        stride = self.stride()
        requires_grad = False
        args = (self._data, storage_offset, size, stride, requires_grad, None, None)
        return (
            _rebuild_from_type_v2, (_rebuild_tensor_v2, type(self), args, None))

    def __hash__(self):
        return hash(id(self))

    def __len__(self):
        if self.shape == ():
            return 1
        return self.shape[0]

    def __repr__(self) -> str:
        data = self._data
        data.data_sync(True)
        return data.__repr__()[:-1] + f', device={self.device})'

    def __format__(self, format_spec):
        return np.ndarray.__format__(self.numpy(), format_spec)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, slices):
        if self.device.type == 'cpu':
            if isinstance(slices, Tensor):
                data = self.numpy()[slices.numpy()]
            else:
                data = self.numpy()[slices]
            
            return Tensor(MSTensor(np.array(data)), device=self.device)
        return mindtorch.tensor_getitem(self, slices)

    def __setitem__(self, slices, value):
        """"""
        if self.device.type == 'cpu':
            data = self.numpy()
            if isinstance(value, Tensor):
                data[slices] = value.numpy()
            else:
                data[slices] = value
            
            return Tensor(MSTensor(np.array(data)), device=self.device)
        else:
            mindtorch.tensor_setitem(self, slices, value)
        return self

    def __add__(self, other):
        return mindtorch.add(self, other)

    def __iadd__(self, other):
        return self.copy_(mindtorch.add(self, other))

    def __radd__(self, other):
        return Tensor.__add__(other, self)

    def __truediv__ (self, other):
        return mindtorch.div(self, other)

    def __rtruediv__ (self, other):
        return mindtorch.div(other, self)

    def __ne__(self, other):
        return mindtorch.ne(self, other)

    def __neg__(self):
        return mindtorch.neg(self)

    def __mul__(self, other):
        return mindtorch.mul(self, other)

    def __rmul__(self, other):
        return mindtorch.mul(other, self)

    def __imul__(self, other):
        return self.copy_(mindtorch.mul(self, other))

    def __pow__(self, other):
        return mindtorch.pow(self, other)

    def __rpow__(self, other):
        return mindtorch.pow(other, self)

    def __sub__(self, other):
        return mindtorch.sub(self, other)

    def __isub__(self, other):
        return self.copy_(mindtorch.sub(self, other))

    def __rsub__(self, other):
        return mindtorch.sub(other, self)

    def __eq__(self, other):
        return mindtorch.equal(self, other)

    def __gt__(self, other):
        return mindtorch.gt(self, other)

    def __ge__(self, other):
        return mindtorch.ge(self, other)

    def __lt__(self, other):
        return mindtorch.lt(self, other)

    def __le__(self, other):
        return mindtorch.le(self, other)

    def __int__(self):
        return int(self.item())

    def __bool__(self):
        return bool(self.item())

    def __index__(self):
        return int(self.item())

    def __and__(self, other):
        return mindtorch.bitwise_and(self, other)

    def __xor__(self, other):
        return mindtorch.bitwise_xor(self, other)

    def __or__(self, other):
        return mindtorch.bitwise_or(self, other)

    def __invert__(self):
        return mindtorch.logical_not(self)

    def __round__(self):
        return mindtorch.round(self)


    # def __getattribute__(self, name):
    #     if name.endswith('_') and not name.endswith('__') and self.is_leaf and self.requires_grad and mindtorch.is_grad_enabled():
    #         raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
    #     return super().__getattribute__(name)

    # Tensor.new_tensor
    def new_tensor(self, data, *, dtype=None):
        if dtype is not None:
            dtype = self.dtype
        return Tensor(data, dtype)

    # Tensor.new_full
    def new_full(self, size, fill_value, *, dtype=None, device=None, requires_grad=False, layout=mindtorch.strided, pin_memory=False):
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return mindtorch.full(size, fill_value, dtype=dtype, device=device, requires_grad=requires_grad)

    # Tensor.new_empty
    def new_empty(self, size, *, dtype=None, device=None, requires_grad=False, layout=mindtorch.strided, pin_memory=False):
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return mindtorch.empty(*size, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory)

    # Tensor.new_ones
    def new_ones(self, *size, dtype=None, device=None, requires_grad=False, layout=mindtorch.strided, pin_memory=False):
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        if isinstance(size[0], tuple):
            size = size[0]
        return mindtorch.ones(*size, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory)

    # Tensor.new_zeros
    def new_zeros(self, size, *, dtype=None, device=None, requires_grad=False, layout=mindtorch.strided, pin_memory=False):
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return mindtorch.zeros(*size, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory)

    # Tensor.ndim
    @property
    def ndim(self):
        return len(self.shape)

    def dim(self):
        return self.ndim

    def ndimension(self):
        return self.ndim

    # Tensor.real
    def real(self):
        return mindtorch.real(self)

    # Tensor.imag
    def imag(self):
        return mindtorch.imag(self)

    # Tensor.nbytes
    @property
    def nbytes(self):
        return self._data._nbytes

    # Tensor.itemsize
    @property
    def itemsize(self):
        return self._data._itemsize

    # Tensor.abs
    def abs(self):
        return mindtorch.abs(self)

    # Tensor.abs_
    def abs_(self):
        return self.copy_(mindtorch.abs(input))

    # Tensor.absolute
    absolute = abs

    # Tensor.absolute_
    absolute_ = abs_

    # Tensor.acos
    def acos(self):
        return mindtorch.acos(self)

    # Tensor.acos_
    def acos_(self):
        return self.copy_(mindtorch.acos(input))

    # Tensor.arccos
    arccos = acos

    # Tensor.arccos_
    arccos_ = acos_

    # Tensor.add
    def add(self, other, *, alpha=1):
        return mindtorch.add(self, other, alpha=alpha)

    # Tensor.add_
    def add_(self, other, *, alpha=1):
        return mindtorch.inplace_add(self, other, alpha=alpha)

    # Tensor.addbmm
    def addbmm(self, batch1, batch2, *, beta=1, alpha=1):
        return mindtorch.addbmm(self, batch1, batch2, beta=beta, alpha=alpha)

    # Tensor.addbmm_
    def addbmm_(self, batch1, batch2, *, beta=1, alpha=1):
        return self.copy_(mindtorch.addbmm(self, batch1, batch2, beta=beta, alpha=alpha))

    # Tensor.addcdiv
    def addcdiv(self, tensor1, tensor2, *, value=1):
        return mindtorch.addcdiv(self, tensor1, tensor2, value=value)

    # Tensor.addcdiv_
    def addcdiv_(self, tensor1, tensor2, *, value=1):
        return self.copy_(mindtorch.addcdiv(self, tensor1, tensor2, value=value))

    # Tensor.addcmul
    def addcmul(self, tensor1, tensor2, *, value=1):
        return mindtorch.addcmul(self, tensor1, tensor2, value=value)

    # Tensor.addcmul_
    def addcmul_(self, tensor1, tensor2, *, value=1):
        return self.copy_(mindtorch.addcmul(self, tensor1, tensor2, value=value))

    # Tensor.addmm
    def addmm(self, mat1, mat2, *, beta=1, alpha=1):
        return mindtorch.addmm(self, mat1, mat2, beta=beta, alpha=alpha)

    # Tensor.addmm_
    def addmm_(self, mat1, mat2, *, beta=1, alpha=1):
        return self.copy_(mindtorch.addmm(self, mat1, mat2, beta=beta, alpha=alpha))

    # Tensor.sspaddmm


    # Tensor.addmv
    def addmv(self, mat, vec, *, beta=1, alpha=1):
        return mindtorch.addmv(self, mat, vec, beta=beta, alpha=alpha)

    # Tensor.addmv_
    def addmv_(self, mat, vec, *, beta=1, alpha=1):
        return self.copy_(mindtorch.addmv(self, mat, vec, beta=beta, alpha=alpha))

    # Tensor.addr

    # Tensor.addr_


    # Tensor.adjoint

    # Tensor.allclose
    def allclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        return mindtorch.allclose(self, other, rtol, atol, equal_nan)

    # Tensor.amax
    def amax(self, dim=None, keepdim=False):
        return mindtorch.amax(self, dim, keepdim)

    # Tensor.amin
    def amin(self, dim=None, keepdim=False):
        return mindtorch.amin(self, dim, keepdim)

    # Tensor.aminmax
    def aminmax(self, dim=None, keepdim=False):
        return mindtorch.aminmax(self, dim=dim, keepdim=keepdim)

    # Tensor.angle


    # Tensor.apply_
    def apply_(self, callable):
        return self.copy_(callable(self))

    # Tensor.argmax
    def argmax(self, dim=None, keepdim=False):
        out = mindtorch.argmax(self, dim, keepdim)
        return out

    # Tensor.argmin
    def argmin(self, dim=None, keepdim=False):
        out = mindtorch.argmin(self, dim, keepdim)
        return out

    # Tensor.argsort
    def argsort(self, dim=-1, descending=False):
        return mindtorch.argsort(self, dim=-1, descending=False)

    # Tensor.argwhere
    def argwhere(self):
        return mindtorch.argwhere(self)

    # Tensor.asin
    def asin(self):
        return mindtorch.asin(self)

    # Tensor.asin_
    def asin_(self):
        return self.copy_(mindtorch.asin(self))

    # Tensor.arcsin
    arcsin = asin

    # Tensor.arcsin_
    arcsin_ = asin_

    # Tensor.as_strided
    def as_strided(self, size, stride, storage_offset=None):
        return mindtorch.as_strided(self, size, stride, storage_offset)

    # Tensor.atan
    def atan(self):
        return mindtorch.atan(self)

    # Tensor.atan_
    def atan_(self):
        return self.copy_(mindtorch.atan(self))

    # Tensor.arctan
    arctan = atan

    # Tensor.arctan_
    arctan_ = atan_

    # Tensor.atan2
    def atan2(self, other):
        return mindtorch.atan2(self, other)

    # Tensor.atan2_
    def atan2_(self, other):
        return self.copy_(mindtorch.atan2(self, other))

    # Tensor.arctan2
    arctan2 = atan2

    # Tensor.arctan2_
    arctan2_ = atan2_

    # Tensor.all
    def all(self, dim=None, keepdim=False):
        return mindtorch.all(self, dim, keepdim)

    # Tensor.any
    def any(self, dim=None, keepdim=False):
        return mindtorch.any(self, dim, keepdim)

    # Tensor.baddbmm
    def baddbmm(self, batch1, batch2, *, beta=1, alpha=1):
        return mindtorch.baddbmm(self, batch1, batch2, beta=beta, alpha=alpha)

    # Tensor.baddbmm_
    def baddbmm_(self, batch1, batch2, *, beta=1, alpha=1):
        return self.copy_(mindtorch.baddbmm(self, batch1, batch2, beta=beta, alpha=alpha))

    # Tensor.bernoulli
    def bernoulli(self, *, generator=None):
        return mindtorch.bernoulli(self, generator=generator)

    # Tensor.bernoulli_
    def bernoulli_(self, *, generator=None):
        return self.copy_(mindtorch.bernoulli(self, generator=generator))

    # Tensor.bfloat16
    def bfloat16(self):
        return self.to(mindtorch.bfloat16)

    # Tensor.bincount
    def bincount(self, weight=None, minlength=0):
        return mindtorch.bincount(self, weight, minlength)

    # Tensor.bitwise_not
    def bitwise_not(self):
        return mindtorch.bitwise_not(self)

    # Tensor.bitwise_not_
    def bitwise_not_(self):
        return self.copy_(mindtorch.bitwise_not(self))

    # Tensor.bitwise_and
    def bitwise_and(self, other):
        return mindtorch.bitwise_and(self, other)

    # Tensor.bitwise_and_
    def bitwise_and_(self, other):
        return self.copy_(mindtorch.bitwise_and(self, other))

    # Tensor.bitwise_or
    def bitwise_or(self, other):
        return mindtorch.bitwise_or(self, other)

    # Tensor.bitwise_or_
    def bitwise_or_(self, other):
        return self.copy_(mindtorch.bitwise_or(self, other))

    # Tensor.bitwise_xor
    def bitwise_xor(self, other):
        return mindtorch.bitwise_xor(self, other)

    # Tensor.bitwise_xor_
    def bitwise_xor_(self, other):
        return self.copy_(mindtorch.bitwise_xor(self, other))

    # Tensor.bitwise_left_shift


    # Tensor.bitwise_left_shift_


    # Tensor.bitwise_right_shift


    # Tensor.bitwise_right_shift_


    # Tensor.bmm
    def bmm(self, batch2):
        return mindtorch.bmm(self, batch2)

    # Tensor.bool
    def bool(self):
        return self.to(mindtorch.bool)

    # Tensor.byte
    def byte(self):
        return self.to(mindtorch.uint8)

    # Tensor.broadcast_to
    def broadcast_to(self, shape):
        return mindtorch.broadcast_to(self, shape)

    # Tensor.cauchy_


    # Tensor.ceil
    def ceil(self):
        return mindtorch.ceil(self)

    # Tensor.ceil_
    def ceil_(self):
        return self.copy_(mindtorch.ceil(self))

    # Tensor.char
    def char(self):
        return self.to(mindtorch.int8)

    # Tensor.cholesky


    # Tensor.cholesky_inverse


    # Tensor.cholesky_solve


    # Tensor.chunk
    def chunk(self, chunks, dim=0):
        return mindtorch.chunk(self, chunks, dim)

    # Tensor.clamp
    def clamp(self, min=None, max=None):
        return mindtorch.clamp(self, min, max)

    # Tensor.clamp_
    def clamp_(self, min=None, max=None):
        return self.copy_(mindtorch.clamp(self, min, max))

    # Tensor.clip
    def clip(self, min=None, max=None):
        return mindtorch.clip(self, min, max)

    # Tensor.clip_
    def clip_(self, min=None, max=None):
        return self.copy_(mindtorch.clip(self, min, max))

    # Tensor.clone
    def clone(self):
        return mindtorch.clone(self)

    # Tensor.contiguous
    def contiguous(self):
        return mindtorch.contiguous(self)

    # Tensor.copy_
    def copy_(self, value):
        if self.dtype != value.dtype:
            value = value.to(self.dtype)
        return mindtorch.inplace_copy(self, value)

    # Tensor.conj
    def conj(self):
        return mindtorch.conj(self)

    # Tensor.conj_physical


    # Tensor.conj_physical_


    # Tensor.resolve_conj


    # Tensor.resolve_neg


    # Tensor.copysign


    # Tensor.copysign_


    # Tensor.cos
    def cos(self):
        return mindtorch.cos(self)

    # Tensor.cos_
    def cos_(self):
        return self.copy_(mindtorch.cos(self))

    # Tensor.cosh
    def cosh(self):
        return mindtorch.cosh(self)

    # Tensor.cosh_
    def cosh_(self):
        return self.copy_(mindtorch.cosh(self))

    # Tensor.corrcoef


    # Tensor.count_nonzero
    def count_nonzero(self, dim=None):
        return mindtorch.count_nonzero(self, dim)

    # Tensor.cov


    # Tensor.acosh
    def acosh(self):
        return mindtorch.acosh(self)

    # Tensor.acosh_
    def acosh_(self):
        return self.copy_(mindtorch.acosh(self))

    # Tensor.arccosh
    arccosh = acosh

    # Tensor.arccosh_
    arccosh_ = acosh_

    # Tensor.cross


    # Tensor.logcumsumexp


    # Tensor.cummax


    # Tensor.cummin


    # Tensor.cumprod


    # Tensor.cumprod_


    # Tensor.cumsum
    def cumsum(self, dim, dtype=None):
        return mindtorch.cumsum(self, dim, dtype)

    # Tensor.cumsum_
    def cumsum_(self, dim, dtype=None):
        return self.copy_(mindtorch.cumsum(self, dim, dtype))

    # Tensor.chalf


    # Tensor.cfloat


    # Tensor.cdouble


    # Tensor.data_ptr
    def data_ptr(self):
        return self._data._data_ptr()

    # Tensor.deg2rad
    def deg2rad(self):
        return mindtorch.deg2rad(self)

    # Tensor.dequantize


    # Tensor.det


    # Tensor.dense_dim


    # Tensor.diag
    def diag(self, diagonal=0):
        return mindtorch.diag(self, diagonal)

    # Tensor.diag_embed


    # Tensor.diagflat


    # Tensor.diagonal
    def diagnoal(self, offset=0, dim1=0, dim2=1):
        return mindtorch.diagonal(self, offset, dim1, dim2)


    # Tensor.diagonal_scatter

    # Tensor.fill_diagonal_


    # Tensor.fmax


    # Tensor.fmin


    # Tensor.diff


    # Tensor.digamma


    # Tensor.digamma_


    # Tensor.dim_order


    # Tensor.dist


    # Tensor.div
    def div(self, other):
        return mindtorch.div(self, other)

    # Tensor.div_
    def div_(self, other):
        return self.copy_(mindtorch.div(self, other))

    # Tensor.divide
    divide = div

    # Tensor.divide_
    divide_ = div_

    # Tensor.dot
    def dot(self, other):
        return mindtorch.dot(self, other)

    # Tensor.double
    def double(self):
        return self.to(mindtorch.float64)

    # Tensor.dsplit


    # Tensor.element_size
    def element_size(self):
        return self._data._itemsize

    # Tensor.eq
    def eq(self, other):
        return mindtorch.eq(self, other)

    # Tensor.eq_
    def eq_(self, other):
        return self.copy_(mindtorch.eq(self, other))

    # Tensor.equal
    def equal(self, other):
        return mindtorch.eq(self, other)

    # Tensor.erf
    def erf(self):
        return mindtorch.erf(self)

    # Tensor.erf_
    def erf_(self):
        return self.copy_(mindtorch.erf(self))

    # Tensor.erfc
    def erfc(self):
        return mindtorch.erfc(self)

    # Tensor.erfc_
    def erfc_(self):
        return self.copy_(mindtorch.erfc(self))

    # Tensor.erfinv
    def erfinv(self):
        return mindtorch.erfinv(self)


    # Tensor.erfinv_
    def erfinv_(self):
        return self.copy_(mindtorch.erfinv(self))

    # Tensor.exp
    def exp(self):
        return mindtorch.exp(self)

    # Tensor.exp_
    def exp_(self):
        return self.copy_(mindtorch.exp(self))


    # Tensor.expm1
    def expm1(self):
        return mindtorch.expm1(self)


    # Tensor.expm1_
    def expm1_(self):
        return self.copy_(mindtorch.expm1(self))


    # Tensor.expand
    def expand(self, *size):
        if len(size) == 1:
            size = size[0]
        return self.broadcast_to(size)

    # Tensor.expand_as
    def expand_as(self, other):
        return self.expand(other.size())

    # Tensor.exponential_


    # Tensor.fix


    # Tensor.fix_


    # Tensor.fill_
    def fill_(self, value):
        mindtorch.inplace_fill(value)
        return self

    # Tensor.flatten
    def flatten(self, start_dim=0, end_dim=-1):
        return mindtorch.flatten(self, start_dim, end_dim)

    # Tensor.flip
    def flip(self, dims):
        return mindtorch.flip(self, dims)

    # Tensor.fliplr


    # Tensor.flipud


    # Tensor.float
    def float(self):
        return self.to(mindtorch.float32)

    # Tensor.float_power
    def float_power(self, exponent):
        return mindtorch.float_power(self, exponent)

    # Tensor.float_power_
    def float_power_(self, exponent):
        return self.copy_(mindtorch.float_power(self, exponent))

    # Tensor.floor
    def floor(self):
        return mindtorch.floor(self)

    # Tensor.floor_
    def floor_(self):
        return self.copy_(mindtorch.floor(self))

    # Tensor.floor_divide
    def floor_divide(self, other):
        return mindtorch.floor_divide(self, other)

    # Tensor.floor_divide_
    def floor_divide_(self, other):
        return self.copy_(mindtorch.floor_divide(self, other))


    # Tensor.fmod
    def fmod(self, other):
        return mindtorch.fmod(self, other)

    # Tensor.fmod_
    def fmod_(self, other):
        return self.copy_(mindtorch.fmod(self, other))

    # Tensor.frac
    def frac(self):
        return mindtorch.frac(self)

    # Tensor.frac_
    def frac_(self):
        return self.copy_(mindtorch.frac(self))


    # Tensor.frexp


    # Tensor.gather
    def gather(self, dim, index):
        return mindtorch.gather(self, dim, index)

    # Tensor.gcd


    # Tensor.gcd_


    # Tensor.ge
    def ge(self, other):
        return mindtorch.ge(self, other)

    # Tensor.ge_
    def ge_(self, other):
        return self.copy_(mindtorch.ge(self, other))

    # Tensor.greater_equal
    greater_equal = ge

    # Tensor.greater_equal_
    greater_equal_ = ge_


    # Tensor.geometric_


    # Tensor.geqrf


    # Tensor.ger


    # Tensor.get_device
    def get_device(self):
        return self.device.index

    # Tensor.gt
    def gt(self, other):
        return mindtorch.gt(self, other)

    # Tensor.gt_
    def gt_(self, other):
        return self.copy_(mindtorch.gt(self, other))

    # Tensor.greater
    greater = gt

    # Tensor.greater_
    greater_ = gt_


    # Tensor.half
    def half(self):
        return self.to(mindtorch.float16)

    # Tensor.hardshrink
    def hardshrink(self, lambd=0.5):
        return mindtorch.nn.functional.hardshrink(self, lambd)

    # Tensor.heaviside


    # Tensor.histc


    # Tensor.histogram


    # Tensor.hsplit


    # Tensor.hypot


    # Tensor.hypot_


    # Tensor.i0


    # Tensor.i0_


    # Tensor.igamma


    # Tensor.igamma_


    # Tensor.igammac


    # Tensor.igammac_


    # Tensor.index_add_
    def index_add_(self, dim, index, source, *, alpha=1):
        return self.copy_(mindtorch.index_add(self, dim, source, alpha=alpha))

    # Tensor.index_add
    def index_add(self, dim, index, source, *, alpha=1):
        return mindtorch.index_add(self, dim, source, alpha=alpha)

    # Tensor.index_copy_


    # Tensor.index_copy


    # Tensor.index_fill_


    # Tensor.index_fill


    # Tensor.index_put_


    # Tensor.index_put


    # Tensor.index_reduce_


    # Tensor.index_reduce

    # Tensor.index_select
    def index_select(self, dim, index):
        return mindtorch.index_select(self, dim, index)

    # Tensor.indices


    # Tensor.inner


    # Tensor.int
    def int(self):
        return self.to(mindtorch.int64)

    # Tensor.int_repr


    # Tensor.inverse


    # Tensor.isclose
    def isclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        return mindtorch.isclose(self, other, rtol, atol, equal_nan)

    # Tensor.isfinite
    def isfinite(self):
        return mindtorch.isfinite(self)

    # Tensor.isinf
    def isinf(self):
        return mindtorch.isinf(self)

    # Tensor.isposinf


    # Tensor.isneginf


    # Tensor.isnan
    def isnan(self):
        return mindtorch.isnan(self)

    # Tensor.is_contiguous
    def is_contiguous(self):
        return self._data.is_contiguous()

    # Tensor.is_complex
    def is_complex(self):
        return False

    # Tensor.is_conj


    # Tensor.is_floating_point
    def is_floating_point(self):
        return isinstance(self.dtype, typing.Float)

    # Tensor.is_inference


    # Tensor.is_leaf
    @property
    def is_leaf(self):
        if not self.requires_grad:
            return True
        if self.requires_grad and self._user_created:
            return True
        return False

    # Tensor.is_pinned


    # Tensor.is_set_to


    # Tensor.is_shared


    # Tensor.is_signed


    # Tensor.is_sparse
    @property
    def is_sparse(self):
        return False

    # Tensor.istft


    # Tensor.isreal


    # Tensor.item
    def item(self):
        return self._data._item()

    # Tensor.kthvalue


    # Tensor.lcm


    # Tensor.lcm_


    # Tensor.ldexp


    # Tensor.ldexp_


    # Tensor.le
    def le(self, other):
        return mindtorch.le(self, other)

    # Tensor.le_
    def le_(self, other):
        return self.copy_(mindtorch.le(self, other))

    # Tensor.less_equal
    less_equal = le

    # Tensor.less_equal_
    less_equal_ = le_


    # Tensor.lerp
    def lerp(self, end, weight):
        return mindtorch.lerp(self, end, weight)

    # Tensor.lerp_
    def lerp_(self, end, weight):
        return self.copy_(mindtorch.lerp(self, end, weight))


    # Tensor.lgamma


    # Tensor.lgamma_


    # Tensor.log
    def log(self):
        return mindtorch.log(self)

    # Tensor.log_
    def log_(self):
        return self.copy_(mindtorch.log(self))

    # Tensor.logdet


    # Tensor.log10
    def log10(self):
        return mindtorch.log10(self)


    # Tensor.log10_
    def log10_(self):
        return self.copy_(mindtorch.log10(self))

    # Tensor.log1p
    def log1p(self):
        return mindtorch.log1p(self)


    # Tensor.log1p_
    def log1p_(self):
        return self.copy_(mindtorch.log1p(self))


    # Tensor.log2
    def log2(self):
        return mindtorch.log2(self)


    # Tensor.log2_
    def log2_(self):
        return self.copy_(mindtorch.log2(self))


    # Tensor.log_normal_


    # Tensor.logaddexp


    # Tensor.logaddexp2


    # Tensor.logsumexp
    def logsumexp(self, dim, keepdim=False):
        return mindtorch.logsumexp(self, dim, keepdim)

    # Tensor.logical_and
    def logical_and(self, other):
        return mindtorch.logical_and(self, other)

    # Tensor.logical_and_
    def logical_and_(self, other):
        return self.copy_(mindtorch.logical_and(self, other))


    # Tensor.logical_not
    def logical_not(self):
        return mindtorch.logical_not(self)


    # Tensor.logical_not_
    def logical_not_(self):
        return self.copy_(mindtorch.logical_not(self))


    # Tensor.logical_or
    def logical_or(self, other):
        return mindtorch.logical_or(self, other)


    # Tensor.logical_or_
    def logical_or_(self, other):
        return self.copy_(mindtorch.logical_or(self, other))


    # Tensor.logical_xor
    def logical_xor(self, other):
        return mindtorch.logical_xor(self, other)

    # Tensor.logical_xor_
    def logical_xor_(self, other):
        return self.copy_(mindtorch.logical_xor(self, other))

    # Tensor.logit


    # Tensor.logit_


    # Tensor.long
    def long(self):
        return self.to(mindtorch.int64)

    # Tensor.lt
    def lt(self, other):
        return mindtorch.lt(self, other)

    # Tensor.lt_
    def lt_(self, other):
        return self.copy_(mindtorch.lt(self, other))

    # Tensor.less
    less = lt

    # Tensor.less_
    less_ = lt_

    # Tensor.lu


    # Tensor.lu_solve


    # Tensor.as_subclass


    # Tensor.map_


    # Tensor.masked_scatter_


    # Tensor.masked_scatter


    # Tensor.masked_fill_
    def masked_fill_(self, mask, value):
        return self.copy_(mindtorch.masked_fill(self, mask, value))

    # Tensor.masked_fill
    def masked_fill(self, mask, value):
        return mindtorch.masked_fill(self, mask, value)

    # Tensor.masked_select
    def masked_select(self, mask):
        return mindtorch.masked_select(self, mask)

    # Tensor.matmul
    def matmul(self, other):
        return mindtorch.matmul(self, other)

    # Tensor.matrix_power


    # Tensor.matrix_exp


    # Tensor.max
    def max(self, dim=None, keepdim=False):
        return mindtorch.max(self, dim, keepdim)

    # Tensor.maximum
    def maximum(self, other):
        return mindtorch.maximum(self, other)

    # Tensor.mean
    def mean(self, dim=None, keepdim=False, *, dtype=None):
        return mindtorch.mean(self, dim, keepdim, dtype=dtype)

    # Tensor.module_load


    # Tensor.nanmean


    # Tensor.median
    def median(self, dim=-1, keepdim=False):
        return mindtorch.median(self, dim, keepdim)

    # Tensor.nanmedian


    # Tensor.min
    def min(self, dim=None, keepdim=False):
        return mindtorch.min(self, dim, keepdim)

    # Tensor.minimum
    def minimum(self, other):
        return mindtorch.minimum(self, other)

    # Tensor.mm
    mm = matmul

    # Tensor.smm


    # Tensor.mode
    def mode(self, dim=None, keepdim=False):
        return mindtorch.mode(self, dim, keepdim)

    # Tensor.movedim
    def movedim(self, source, destination):
        return mindtorch.movedim(source, destination)

    # Tensor.moveaxis
    moveaxis = movedim

    # Tensor.msort
    def msort(self):
        return mindtorch.msort(self)

    # Tensor.mul
    def mul(self, other):
        return mindtorch.mul(self, other)

    # Tensor.mul_
    def mul_(self, other):
        return self.copy_(mindtorch.mul(self, other))

    # Tensor.multiply
    multiply = mul

    # Tensor.multiply_
    multiply_ = mul_


    # Tensor.multinomial
    def multinomial(self, num_samples, replacement=False, *, generator=None):
        return mindtorch.multinomial(self, num_samples, replacement, generator=generator)

    # Tensor.mv


    # Tensor.mvlgamma


    # Tensor.mvlgamma_


    # Tensor.nansum
    def nansum(self, dim=None, keepdim=False, *, dtype=None):
        return mindtorch.nansum(self, dim, keepdim, dtype=dtype)

    # Tensor.narrow
    def narrow(self, dim, start, length):
        return mindtorch.narrow(self, dim, start, length)

    # Tensor.narrow_copy
    def narrow_copy(self, dimension, start, length):
        return mindtorch.narrow(self, dimension, start, length).clone()

    # Tensor.nan_to_num
    def nan_to_num(self, nan=0.0, posinf=None, neginf=None):
        return mindtorch.nan_to_num(self, nan, posinf, neginf)

    # Tensor.nan_to_num_
    def nan_to_num_(self, nan=0.0, posinf=None, neginf=None):
        return self.copy_(mindtorch.nan_to_num(self, nan, posinf, neginf))

    # Tensor.ne
    def ne(self, other):
        return mindtorch.ne(self, other)

    # Tensor.ne_
    def ne_(self, other):
        return self.copy_(mindtorch.ne(self, other))

    # Tensor.not_equal
    not_equal = ne

    # Tensor.not_equal_
    not_equal_ = ne_


    # Tensor.neg
    def neg(self):
        return mindtorch.neg(self)

    # Tensor.neg_
    def neg_(self):
        return self.copy_(mindtorch.neg(self))

    # Tensor.negative
    negative = neg

    # Tensor.negative_
    negative_ = neg_


    # Tensor.numel
    def numel(self):
        return self._data._size

    # Tensor.nelement
    nelement = numel

    # Tensor.nextafter


    # Tensor.nextafter_


    # Tensor.nonzero
    def nonzero(self):
        return mindtorch.nonzero(self)

    # Tensor.norm
    def norm(self, p='fro', dim=None, keepdim=False, dtype=None):
        return mindtorch.norm(self, p, dim, keepdim, dtype)

    # Tensor.normal_
    def normal_(self, mean=0, std=1, *, generator=None):
        return mindtorch.inplace_normal(self, mean, std, generator=generator)

    # Tensor.numpy
    def numpy(self):
        return self._data.asnumpy()

    def mindspore(self):
        return mindspore.Tensor(self._data)

    # Tensor.orgqr


    # Tensor.ormqr


    # Tensor.outer
    def outer(self, vec2):
        return mindtorch.outer(self, vec2)

    # Tensor.permute
    def permute(self, *dims):
        return mindtorch.permute(self, dims)

    # Tensor.pin_memory


    # Tensor.pinverse


    # Tensor.polygamma


    # Tensor.polygamma_


    # Tensor.positive
    def positive(self):
        return self

    # Tensor.pow
    def pow(self, exponent):
        return mindtorch.pow(self, exponent)

    # Tensor.pow_
    def pow_(self, exponent):
        return self.copy_(mindtorch.pow(self, exponent))


    # Tensor.prod
    def prod(self, dim=None, keepdim=False, dtype=None):
        return mindtorch.prod(self, dim, keepdim, dtype=dtype)

    # Tensor.put_


    # Tensor.qr


    # Tensor.qscheme


    # Tensor.quantile


    # Tensor.nanquantile


    # Tensor.q_scale


    # Tensor.q_zero_point


    # Tensor.q_per_channel_scales


    # Tensor.q_per_channel_zero_points


    # Tensor.q_per_channel_axis


    # Tensor.rad2deg


    # Tensor.ravel


    # Tensor.reciprocal
    def reciprocal(self):
        return mindtorch.reciprocal(self)

    # Tensor.reciprocal_
    def reciprocal_(self):
        return self.copy_(mindtorch.reciprocal(self))


    # Tensor.record_stream


    # Tensor.register_hook
    def register_hook(self, hook):
        return self._data.register_hook(hook)

    # Tensor.register_post_accumulate_grad_hook


    # Tensor.remainder
    def remainder(self, other):
        return mindtorch.remainder(self, other)

    # Tensor.remainder_
    def remainder_(self, other):
        return self.copy_(mindtorch.remainder(self, other))

    # Tensor.renorm


    # Tensor.renorm_


    # Tensor.repeat
    def repeat(self, *repeats):
        return mindtorch.tile(self, repeats)

    # Tensor.repeat_interleave
    def repeat_interleave(self, repeats, dim=None):
        return mindtorch.repeat_interleave(self, repeats, dim)

    # Tensor.reshape
    def reshape(self, *shape):
        return mindtorch.reshape(self, *shape)

    # Tensor.reshape_as
    def reshape_as(self, other):
        return self.reshape(*other.shape)

    # Tensor.resize_
    def resize_(self, *shape):
        self.data = mindtorch.reshape(self, *shape)
        return self

    # Tensor.resize_as_
    def resize_as_(self, other):
        self.data = mindtorch.reshape(self, *other.shape)
        return self

    # Tensor.retains_grad
    @property
    def retains_grad(self):
        return not self.is_leaf and self._retain_grad

    # Tensor.roll
    def roll(self, shifts, dims=None):
        return mindtorch.roll(self, shifts, dims)

    # Tensor.rot90


    # Tensor.round
    def round(self):
        return mindtorch.round(self)

    # Tensor.round_
    def round_(self):
        return self.copy_(mindtorch.round(self))


    # Tensor.rsqrt
    def rsqrt(self):
        return mindtorch.rsqrt(self)

    # Tensor.rsqrt_
    def rsqrt_(self):
        return self.copy_(mindtorch.rsqrt(self))


    # Tensor.scatter
    def scatter(self, dim, index, src):
        return mindtorch.scatter(self, dim, index, src)

    # Tensor.scatter_
    def scatter_(self, dim, index, src):
        return mindtorch.inplace_scatter(self, dim, index, src)

    # Tensor.scatter_add_
    def scatter_add_(self, dim, index, src):
        return self.copy_(mindtorch.scatter_add(self, dim, index, src))

    # Tensor.scatter_add
    def scatter_add(self, dim, index, src):
        return mindtorch.scatter_add(self, dim, index, src)


    # Tensor.scatter_reduce_
    def scatter_reduce_(self, dim, index, src):
        return self.copy_(mindtorch.scatter_reduce(self, dim, index, src))


    # Tensor.scatter_reduce
    def scatter_reduce(self, dim, index, src):
        return mindtorch.scatter_reduce(self, dim, index, src)


    # Tensor.select
    def select(self, dim, index):
        return mindtorch.select(self, dim, index)

    # Tensor.select_scatter


    # Tensor.set_


    # Tensor.share_memory_


    # Tensor.short
    def short(self):
        return self.to(mindtorch.int16)

    # Tensor.sigmoid
    def sigmoid(self):
        return mindtorch.sigmoid(self)

    # Tensor.sigmoid_
    def sigmoid_(self):
        return self.copy_(mindtorch.sigmoid(self))

    # Tensor.sign
    def sign(self):
        return mindtorch.sign(self)

    # Tensor.sign_
    def sign_(self):
        return self.copy_(mindtorch.sign(self))


    # Tensor.signbit


    # Tensor.sgn


    # Tensor.sgn_


    # Tensor.sin
    def sin(self):
        return mindtorch.sin(self)

    # Tensor.sin_
    def sin_(self):
        return self.copy_(mindtorch.sin(self))


    # Tensor.sinc
    def sinc(self):
        return mindtorch.sinc(self)


    # Tensor.sinc_
    def sinc_(self):
        return self.copy_(mindtorch.sinc(self))

    # Tensor.sinh
    def sinh(self):
        return mindtorch.sinh(self)


    # Tensor.sinh_
    def sinh_(self):
        return self.copy_(mindtorch.sinh(self))


    # Tensor.asinh
    def asinh(self):
        return mindtorch.asinh(self)


    # Tensor.asinh_
    def asinh_(self):
        return self.copy_(mindtorch.asinh(self))


    # Tensor.arcsinh
    arcsinh_ = asinh

    # Tensor.arcsinh_
    arcsinh_ = asinh_


    # Tensor.size
    def size(self, dim=None):
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    # Tensor.slogdet


    # Tensor.slice_scatter


    # Tensor.softmax
    def softmax(self, dim, dtype=None):
        return mindtorch.softmax(self, dim, dtype=dtype)

    # Tensor.sort
    def sort(self, dim=-1, descending=False):
        return mindtorch.sort(self, dim=dim, descending=descending)

    # Tensor.split
    def split(self, split_size, dim=0):
        return mindtorch.split(self, split_size, dim)

    # Tensor.sparse_mask


    # Tensor.sparse_dim


    # Tensor.sqrt
    def sqrt(self):
        return mindtorch.sqrt(self)

    # Tensor.sqrt_
    def sqrt_(self):
        return self.copy_(mindtorch.sqrt(self))


    # Tensor.square
    def square(self):
        return mindtorch.square(self)


    # Tensor.square_
    def square_(self):
        return self.copy_(mindtorch.square(self))

    # Tensor.squeeze
    def squeeze(self, dim):
        return mindtorch.squeeze(self, dim)

    # Tensor.squeeze_
    def squeeze_(self, dim):
        return self.copy_(mindtorch.squeeze(self, dim))


    # Tensor.std
    def std(self, dim=None, *, correction=1, keepdim=False):
        return mindtorch.std(self, dim, correction=correction, keepdim=keepdim)

    # Tensor.stft


    # Tensor.storage


    # Tensor.untyped_storage


    # Tensor.storage_offset


    # Tensor.storage_type


    # Tensor.stride
    def stride(self, dim=None):
        if dim is None:
            return self._data.stride()
        return self._data.stride()[dim]


    # Tensor.sub
    def sub(self, other, *, alpha=1):
        return mindtorch.sub(self, other, alpha=alpha)

    # Tensor.sub_
    def sub_(self, other, *, alpha=1):
        return self.copy_(mindtorch.sub(self, other, alpha=alpha))


    # Tensor.subtract
    subtract = sub

    # Tensor.subtract_
    subtract_ = sub_

    # Tensor.sum
    def sum(self, dim=None, keepdim=False, dtype=None):
        return mindtorch.sum(self, dim, keepdim, dtype=dtype)

    # Tensor.sum_to_size


    # Tensor.svd


    # Tensor.swapaxes
    def swapaxes(self, dim0, dim1):
        return mindtorch.swapaxes(self, dim0, dim1)

    # Tensor.swapdims
    swapdims = swapaxes

    # Tensor.t
    def t(self):
        return mindtorch.t(self)

    # Tensor.t_
    def t_(self):
        self.data = mindtorch.t(self)
        return self

    # Tensor.tensor_split


    # Tensor.tile
    def tile(self, dims):
        return mindtorch.tile(self, dims)

    # Tensor.to
    def _move_to(self, device, non_blocking=False):
        if self.device == device:
            return self
        else:
            device_str = device_map[device.type]
            self._data.data_sync(True)
            data = self._data.move_to(device_str, blocking=not non_blocking)
            out = Tensor(data, device=device)
            out.requires_grad_(self.requires_grad)
            return out

    def to(self, *args, **kwargs):
        non_blocking = kwargs.get('non_blocking', False)
        copy = kwargs.get('copy', False)
        out = self
        for arg in args:
            if isinstance(arg, device_):
                out = Tensor._move_to(out, arg, non_blocking)
            elif isinstance(arg, str):
                device = device_(arg)
                out = Tensor._move_to(out, device, non_blocking)
            elif isinstance(arg, mindspore.common.dtype.Type):
                if out.dtype == arg:
                    return out
                else:
                    out = mindtorch.cast(out, arg)
            elif isinstance(arg, Tensor):
                out = Tensor._move_to(out, arg.device, non_blocking)
                if out.dtype == arg:
                    return out
                else:
                    out = mindtorch.cast(out, arg)
        return out

    # Tensor.take
    def take(self, index):
        return mindtorch.take(self, index)

    # Tensor.take_along_dim


    # Tensor.tan
    def tan(self):
        return mindtorch.tan(self)

    # Tensor.tan_
    def tan_(self):
        return self.copy_(mindtorch.tan(self))


    # Tensor.tanh
    def tanh(self):
        return mindtorch.tanh(self)


    # Tensor.tanh_
    def tanh_(self):
        return self.copy_(mindtorch.tanh(self))


    # Tensor.atanh

    def atanh(self):
        return mindtorch.atanh(self)


    # Tensor.atanh_
    def atanh_(self):
        return self.copy_(mindtorch.atanh(self))


    # Tensor.arctanh
    arctanh = atanh

    # Tensor.arctanh_
    arctanh_ = atanh_

    # Tensor.tolist
    def tolist(self):
        return self.numpy().tolist()

    # Tensor.topk
    def topk(self, k, dim=None, largest=True, sorted=True):
        return mindtorch.topk(self, k, dim, largest, sorted)

    # Tensor.to_dense


    # Tensor.to_sparse


    # Tensor.to_sparse_csr


    # Tensor.to_sparse_csc


    # Tensor.to_sparse_bsr


    # Tensor.to_sparse_bsc


    # Tensor.trace


    # Tensor.transpose
    def transpose(self, dim0, dim1):
        return mindtorch.transpose(self, dim0, dim1)

    # Tensor.transpose_
    def transpose_(self, dim0, dim1):
        self.data = mindtorch.transpose(self, dim0, dim1)
        return self

    # Tensor.triangular_solve


    # Tensor.tril
    def tril(self, diagonal=0):
        return mindtorch.tril(self, diagonal)

    # Tensor.tril_
    def tril_(self, diagonal=0):
        return self.copy_(mindtorch.tril(self, diagonal))


    # Tensor.triu
    def triu(self, diagonal=0):
        return mindtorch.triu(self, diagonal)


    # Tensor.triu_
    def triu_(self, diagonal=0):
        return self.copy_(mindtorch.triu(self, diagonal))


    # Tensor.true_divide
    def true_divide(self, other):
        return mindtorch.true_divide(self, other)

    # Tensor.true_divide_
    def true_divide_(self, other):
        return self.copy_(mindtorch.true_divide(self, other))


    # Tensor.trunc
    def trunc(self):
        return mindtorch.trunc(self)

    # Tensor.trunc_
    def trunc_(self):
        return self.copy_(mindtorch.trunc(self))


    # Tensor.type
    def type(self, dtype=None, non_blocking=False):
        if dtype is None:
            dtype_str = str(dtype_class_map[self.dtype])[8:-2]
            dtype_str = dtype_str.replace('_tensor', self.device.type) \
                if self.device.type != 'cpu' else dtype_str.replace('._tensor', '')
            return dtype_str
        return self.to(dtype, non_blocking=non_blocking)

    # Tensor.type_as
    def type_as(self, tensor):
        return self.type(tensor.dtype)

    # Tensor.unbind
    def unbind(self, dim=0):
        return mindtorch.unbind(self, dim)

    # Tensor.unflatten
    def unflatten(self, dim, sizes):
        return mindtorch.unflatten(self, dim, sizes)

    # Tensor.unfold
    def unfold(self, dimension, size, step):
        _indices, _dimension = mindtorch.utils._get_unfold_indices(self.shape, dimension, size, step)
        output = mindtorch.gather(self, _dimension, _indices)
        return mindtorch.transpose(output, _dimension + 1, -1)

    # Tensor.uniform_
    def uniform_(self, *args, **kwargs):
        return mindtorch.inplace_uniform(self, *args, **kwargs)

    # Tensor.random_
    def random_(self, *args, **kwargs):
        if len(args) == 1:
            from_ = args[0]
        elif len(args) == 2:
            from_ = args[0]
            to_ = args[1]

        from_ = kwargs.get("from", 0)
        to_ = kwargs.get("to", None)
        generator_ = kwargs.get("generator", None)
        if not to_:
            if self.dtype == mindtorch.float64:
                return self.uniform_(from_, kMantissaFloat64, generator_)
            elif self.dtype == mindtorch.float32:
                return self.uniform_(from_, kMantissaFloat32, generator_)
            elif self.dtype == mindtorch.float16:
                return self.uniform_(from_, kMantissaFloat16, generator_)
            elif self.dtype == mindtorch.uint8:
                return self.uniform_(from_, kMaxUint8, generator_)
            elif self.dtype == mindtorch.int64:
                return self.uniform_(from_, kMaxInt64, generator_)
            elif self.dtype == mindtorch.int32:
                return self.uniform_(from_, kMaxInt32, generator_)
            elif self.dtype == mindtorch.int16:
                return self.uniform_(from_, kMaxInt16, generator_)
            elif self.dtype == mindtorch.int8:
                return self.uniform_(from_, kMaxInt8, generator_)
        to_ = to_ - 1 if to_ > 1 else to_
        return self.uniform_(from_, to_, generator_)

    # Tensor.unique
    def unique(self, sorted=True, return_inverse=False, return_counts=False, dim=None):
        return mindtorch.unique(self, sorted, return_inverse, return_counts, dim)

    # Tensor.unique_consecutive
    def unique_consecutive(self, return_inverse=False, return_counts=False, dim=None):
        return mindtorch.unique_consecutive(self, return_inverse, return_counts, dim)

    # Tensor.unsqueeze
    def unsqueeze(self, dim):
        return mindtorch.unsqueeze(self, dim)

    # Tensor.unsqueeze_
    def unsqueeze_(self, dim):
        return self.copy_(mindtorch.unsqueeze(self, dim))


    # Tensor.values


    # Tensor.var
    def var(self, dim=None, *, correction=1, keepdim=False):
        return mindtorch.var(self, dim, correction=correction, keepdim=keepdim)

    # Tensor.vdot


    # Tensor.view
    def view(self, *shape):
        return self.reshape(*shape)

    # Tensor.view_as
    def view_as(self, other):
        return self.reshape(*other.shape)

    # Tensor.vsplit


    # Tensor.where
    def where(self, condition, y):
        return mindtorch.where(condition, self, y)

    # Tensor.xlogy
    def xlogy(self, other):
        return mindtorch.xlogy(self, other)

    # Tensor.xlogy_
    def xlogy_(self, other):
        return self.copy_(mindtorch.xlogy(self, other))

    # Tensor.zero_
    def zero_(self):
        return mindtorch.inplace_zero(self) 

    # Tensor.detach
    def detach(self):
        out = self.data
        out._requires_grad = False
        return out

    # Tensor.detach_
    def detach_(self):
        self.requires_grad_(self)
        return self

    def stub_sync(self):
        if self.stub:
            self.tensor = self.stub.get_value()
            self.stub = None
        return self.tensor

def tensor(data, *, dtype=None, device=None, requires_grad=False):
    if isinstance(data, Tensor):
        UserWarning("To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than mindtorch.tensor(sourceTensor).")
        return Tensor(data)
    data = MSTensor(data, dtype=dtype)
    tensor = Tensor(data).to(device)
    tensor._user_created = True
    tensor.requires_grad_(requires_grad)
    return tensor

def is_tensor(x):
    return isinstance(x, Tensor)

class FloatTensor(Tensor):
    def __init__(self, *input, device=None):
        super().__init__(*input, device, mindspore.float32)

class DoubleTensor(Tensor):
    def __init__(self, *input, device=None):
        super().__init__(*input, device, mindspore.float64)

class HalfTensor(Tensor):
    def __init__(self, *input, device=None):
        super().__init__(*input, device, mindspore.float16)

class BFloat16Tensor(Tensor):
    def __init__(self, *input, device=None):
        super().__init__(*input, device, mindspore.bfloat16)

class IntTensor(Tensor):
    def __init__(self, *input, device=None):
        super().__init__(*input, device, mindspore.int32)

class LongTensor(Tensor):
    def __init__(self, *input, device=None):
        super().__init__(*input, device, mindspore.int64)

class BoolTensor(Tensor):
    def __init__(self, *input, device=None):
        super().__init__(*input, device, mindspore.bool_)

class ByteTensor(Tensor):
    def __init__(self, *input, device=None):
        super().__init__(*input, device, mindspore.uint8)


dtype_class_map = {
    mindspore.float32: FloatTensor,
    mindspore.float16: HalfTensor,
    mindspore.bfloat16: BFloat16Tensor,
    mindspore.float64: DoubleTensor,
    mindspore.int32: IntTensor,
    mindspore.int64: LongTensor,
    mindspore.bool_: BoolTensor,
    mindspore.uint8: ByteTensor
}

def _rebuild_from_type_v2(func, new_type, args, state):
    ret = func(*args)
    return ret


__all__ = ['tensor', 'is_tensor', 'Tensor']