import numpy as np

from mindspore._c_expression import typing
from mindspore._c_expression.typing import Type

dtype = Type

# type definition
bool = typing.kBool

qint4x2 = typing.kInt4
int8 = typing.kInt8
byte = int8
int16 = typing.kInt16
short = int16
int32 = typing.kInt32
intc = int32
int64 = typing.kInt64
intp = int64

uint8 = typing.kUInt8
ubyte = uint8
uint16 = typing.kUInt16
ushort = uint16
uint32 = typing.kUInt32
uintc = uint32
uint64 = typing.kUInt64
uintp = uint64

float16 = typing.kFloat16
half = float16
float32 = typing.kFloat32
single = float32
float64 = typing.kFloat64
double = float64
bfloat16 = typing.kBFloat16
complex64 = typing.kComplex64
complex128 = typing.kComplex128

float = float32
long = int64
int = int32
cfloat = complex64
cdouble = complex128

np2dtype = {
    np.bool_: bool,
    np.int8: int8,
    np.int16: int16,
    np.int32: int32,
    np.int64: int64,
    np.uint8: uint8,
    np.uint16: uint16,
    np.uint32: uint32,
    np.uint64: uint64,
    np.float16: float16,
    np.float32: float32,
    np.float64: float64,
}

dtype2np = {
    bool    : np.bool_,
    int8    : np.int8,
    int16   : np.int16,
    int32   : np.int32,
    int64   : np.int64,
    uint8   : np.uint8,
    uint16  : np.uint16,
    uint32  : np.uint32,
    uint64  : np.uint64,
    float16 : np.float16,
    float32 : np.float32,
    float64 : np.float64,
}