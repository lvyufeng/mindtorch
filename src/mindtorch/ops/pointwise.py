"""pointwise op"""

import math
import numbers
from mindspore.ops.auto_generate.gen_arg_handler import str_to_enum

import mindtorch
from mindtorch.executor import execute


# abs
def abs(input):
    return execute("abs", input)


# absolute
def absolute(input):
    return abs(input)


# acos
def acos(input):
    return execute("acos", input)


# arccos
def arrcos(input):
    return acos(input)


# acosh
def acosh(input):
    return execute("acosh_ext", input)


# arccosh
def arccosh(input):
    return acosh(input)


# add
def add(input, other, *, alpha=1):
    if isinstance(input, mindtorch.Tensor):
        device = input.device
    else:
        device = other.device
    return execute("add_ext", input, other, alpha, device=device)


# addcdiv
def addcdiv(input, tensor1, tensor2, *, value=1):
    return execute("addcdiv", input, tensor1, tensor2, value)


# addcmul
def addcmul(input, tensor1, tensor2, *, value=1):
    return execute("addcmul", input, tensor1, tensor2, value)


# angle
def angle(input):
    return execute("angle", input)


# asin
def asin(input):
    return execute("asin_ext", input)


# arcsin
def arcsin(input):
    return asin(input)


# asinh
def asinh(input):
    return execute("asinh_ext", input)


# arcsinh
def arcsinh(input):
    return asinh(input)


# atan
def atan(input):
    return execute("atan_ext", input)


# arctan
def arctan(input):
    return atan(input)


# atanh
def atanh(input):
    return execute("atanh", input)


# arctanh
def arctanh(input):
    return atanh(input)


# atan2
def atan2(input, other):
    return execute("atan2_ext", input, other)


# arctan2
def arctan2(input, other):
    return atan2(input, other)


# bitwise_not
def bitwise_not(input, *, out=None):
    output = execute("bitwise_not", input)
    if out is None:
        return output
    out.data = output
    return out

# bitwise_and
def bitwise_and(input, other):
    if not isinstance(other, mindtorch.Tensor):
        return execute("bitwise_and_scalar", input, other)
    return execute("bitwise_and_tensor", input, other)


# bitwise_or
def bitwise_or(input, other):
    if not isinstance(other, mindtorch.Tensor):
        return execute("bitwise_or_scalar", input, other)
    return execute("bitwise_or_tensor", input, other)


# bitwise_xor
def bitwise_xor(input, other):
    if not isinstance(other, mindtorch.Tensor):
        return execute("bitwise_xor_scalar", input, other)
    return execute("bitwise_xor_tensor", input, other)


# bitwise_left_shift


# bitwise_right_shift


# ceil
def ceil(input):
    return execute("ceil", input)


# clamp
def clamp(input, min=None, max=None):
    if isinstance(min, mindtorch.Tensor) or isinstance(max, mindtorch.Tensor):
        return execute("clamp_tensor", input, min, max)
    return execute("clamp_scalar", input, min, max)


# clip
def clip(input, min=None, max=None):
    return clamp(input, min, max)


# conj_physical


# copysign


# cos
def cos(input):
    return execute("cos", input)


# cosh
def cosh(input):
    return execute("cosh", input)


# deg2rad
def deg2rad(input):
    return input * math.pi / 180.0


# div
def div(input, other, *, rounding_mode=None):
    if rounding_mode is not None and rounding_mode not in ["floor", "trunc"]:
        raise ValueError(
            "For ops.div, rounding_mode value should be None, 'floor' or 'trunc'."
        )
    if isinstance(input, mindtorch.Tensor):
        device = input.device
    else:
        device = other.device
        
    if rounding_mode:
        output = execute(
            "divmod",
            input,
            other,
            input,
            other,
            (
                rounding_mode
                if rounding_mode is None
                else str_to_enum("DivMod", "rounding_mode", rounding_mode)
            ),
            device=device
        )
    else:
        output = execute("div", input, other, device=device)
    return output


# divide
def divide(input, other):
    return div(input, other)


# digamma


# erf
def erf(input):
    return execute("erf", input)


# erfc
def erfc(input):
    return execute("erfc", input)


# erfinv
def erfinv(input):
    return execute("erfinv", input)


# exp
def exp(input, out=None):
    output = execute("exp", input)
    if out is not None:
        out.data = output
    else:
        return output


# exp2
def exp2(input):
    return execute("exp2", input)


# expm1
def expm1(input):
    return execute("expm1", input)


# fake_quantize_per_channel_affine


# fake_quantize_per_tensor_affine


# fix


# float_power
def float_power(input, exponent):
    if isinstance(input, mindtorch.Tensor) and isinstance(exponent, numbers.Number):
        return execute("pow_tensor_scalar", input, exponent)
    if isinstance(input, numbers.Number) and isinstance(exponent, mindtorch.Tensor):
        return execute("pow_scalar_tensor", input, exponent)

    return pow(input, exponent)


# floor
def floor(input):
    return execute("floor", input)


# floor_divide
def floor_divide(input, other):
    return execute("floor_div", input, other)


# fmod
def fmod(input, other):
    if isinstance(input, mindtorch.Tensor) and isinstance(other, numbers.Number):
        return execute("fmod_scalar", input, other)
    if isinstance(input, numbers.Number) and isinstance(other, mindtorch.Tensor):
        return execute("fmod_scalar", input, other)
    return execute("fmod_tensor", input, other)


# frac
def frac(input):
    return execute("frac", input)


# frexp


# imag


# ldexp


# lerp
def lerp(input, end, weight):
    return execute("lerp", input, end, weight)


# lgamma


# log
def log(input):
    return execute("log", input)


# log10


# log1p
def log1p(input):
    return execute("log1p", input)


# log2
def log2(input):
    return execute("log2", input)


# logaddexp
def logaddexp(input, other):
    return execute("logaddexp", input, other)


# logaddexp2


# logical_and
def logical_and(input, other):
    return execute("logical_and", input, other)


# logical_not
def logical_not(input):
    return execute("logical_not", input)


# logical_or
def logical_or(input, other):
    return execute("logical_or", input, other)


# logical_xor
def logical_xor(input, other):
    return execute("logical_xor", input, other)


# logit


# hypot


# i0


# igamma


# igammac


# mul
def mul(input, other):
    # if isinstance(other, (float, int, bool)) and isinstance(input, mindtorch.Tensor):
    #     return execute("muls", input, other)
    return execute("mul", input, other)


# multiply
def multiply(input, other):
    return mul(input, other)


# mvlgamma


# nan_to_num
def nan_to_num(input, nan=0.0, posinf=None, neginf=None):
    return execute("nan_to_num", input, nan, posinf, neginf)


# neg
def neg(input):
    return execute("neg", input)


# negative
def negative(input):
    return neg(input)


# nextafter
def nextafter(input, other):
    return execute("next_after", input, other)


# polygamma


# positive
def positive(input):
    return input


# pow
def pow(input, exponent):
    if isinstance(input, mindtorch.Tensor) and isinstance(exponent, numbers.Number):
        return execute("pow_tensor_scalar", input, exponent, device=input.device)
    if isinstance(input, numbers.Number) and isinstance(exponent, mindtorch.Tensor):
        return execute("pow_scalar_tensor", input, exponent, device=exponent.device)
    return execute("pow", input, exponent)


# quantized_batch_norm


# quantized_max_pool1d


# quantized_max_pool2d


# rad2deg


# real


# reciprocal
def reciprocal(input):
    return execute("reciprocal", input)


# remainder
def remainder(input, other):
    if isinstance(input, mindtorch.Tensor) and isinstance(other, numbers.Number):
        return execute("remainder_tensor_scalar", input, other)
    if isinstance(input, numbers.Number) and isinstance(other, mindtorch.Tensor):
        return execute("remainder_scalar_tensor", input, other)
    return execute("remainder_tensor_tensor", input, other)


# round
def round(input):
    return execute("round", input)


# rsqrt
def rsqrt(input):
    return execute("input", input)


# sigmoid
def sigmoid(input):
    return execute("sigmoid", input)


# sign
def sign(input):
    return execute("sign", input)


# sgn

# signbit


# sin
def sin(input):
    return execute("sin", input)


# sinc
def sinc(input):
    return execute("sinc", input)


# sinh
def sinh(input):
    return execute("sinh", input)


# softmax
def softmax(input, dim=-1, *, dtype=None):
    return execute("softmax", input, dim)


# sqrt
def sqrt(input):
    return execute("sqrt", input)


# square
def square(input):
    return execute("square", input)


# sub
def sub(input, other, *, alpha=1, out=None):
    if isinstance(input, mindtorch.Tensor):
        device = input.device
    else:
        device = other.device
    if device.type == 'cpu':
        output = execute("sub", input, alpha * other, device=device)
    output = execute("sub_ext", input, other, alpha, device=device)
    if out is None:
        return output
    out.copy_(output)
    return out

# subtract
def subtract(input, other, *, alpha=1, out=None):
    return sub(input, other, alpha=alpha, out=out)


# tan
def tan(input):
    return execute("tan", input)


# tanh
def tanh(input):
    return execute("tanh", input)


# true_divide
def true_divide(input, other):
    return div(input, other)


# trunc
def trunc(input):
    return execute("trunc", input)


# xlogy
def xlogy(input, other):
    if isinstance(input, mindtorch.Tensor) and isinstance(other, mindtorch.Tensor):
        return execute("xlogy", input, other)
    if isinstance(input, mindtorch.Tensor) and isinstance(other, (float, int, bool)):
        return execute("xlogy_scalar_other", input, other)
    if isinstance(input, (float, int, bool)) and isinstance(other, mindtorch.Tensor):
        return execute("xlogy_scalar_self", input, other)
    raise TypeError(f"For 'xlogy', at least one of input and other should be Tensor.")


__all__ = [
    "abs",
    "absolute",
    "acos",
    "acosh",
    "add",
    "addcdiv",
    "addcmul",
    "angle",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "arrcos",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_not",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "ceil",
    "clamp",
    "clip",
    "cos",
    "cosh",
    "deg2rad",
    "div",
    "divide",
    "erf",
    "erfc",
    "erfinv",
    "exp",
    "exp2",
    "expm1",
    "float_power",
    "floor",
    "floor_divide",
    "fmod",
    "frac",
    "lerp",
    "log",
    "log1p",
    "log2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "mul",
    "multiply",
    "nan_to_num",
    "neg",
    "negative",
    "nextafter",
    "positive",
    "pow",
    "reciprocal",
    "remainder",
    "round",
    "rsqrt",
    "sigmoid",
    "sign",
    "sin",
    "sinc",
    "sinh",
    "softmax",
    "sqrt",
    "square",
    "sub",
    "subtract",
    "tan",
    "tanh",
    "true_divide",
    "trunc",
    "xlogy",
]
