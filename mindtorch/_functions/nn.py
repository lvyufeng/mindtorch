from mindtorch import tensor
from mindtorch.autograd import Function, Context
from mindtorch._operations import raw_relu, raw_relu_grad, raw_softmax_crossentropy, raw_mul, \
    raw_softmax_crossentropy_ascend, raw_matmul, raw_add, raw_conv2d, raw_conv2d_gx, raw_conv2d_gw, \
    raw_bias_add, raw_bias_add_grad, raw_dropout, raw_dropout_grad, raw_maxpool, raw_maxpool_grad, \
    raw_nll_loss, raw_nll_loss_grad, raw_layer_norm, raw_layer_norm_grad, raw_gelu, raw_gelu_grad, \
    raw_fold, raw_unfold, raw_softmax, fused_linear, fused_linear_grad, fused_gelu_erf, fused_gelu_erf_grad, \
    fused_softmax_grad, fused_dropout, fused_dropout_grad, raw_linear, raw_linear_grad

from .math import matmul
from .creation import zeros_like
from .utils import sum_to

# =============================================================================
# activation function: sigmoid / relu / softmax / log_softmax / leaky_relu
# =============================================================================
class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, x):
        y = raw_relu(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        y, = ctx.saved_tensors
        gx = raw_relu_grad(gy.data, y)
        return tensor(gx)

class GELU(Function):
    @staticmethod
    def forward(ctx: Context, x):
        y = raw_gelu(x)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x, y, = ctx.saved_tensors
        gx = raw_gelu_grad(gy.data, x, y)
        return tensor(gx)

class GELUErf(Function):
    @staticmethod
    def forward(ctx: Context, x):
        y = fused_gelu_erf(x)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x, = ctx.inputs
        gx = fused_gelu_erf_grad(x.data, gy.data)
        return tensor(gx)

class Softmax(Function):
    @staticmethod
    def forward(ctx: Context, input, axis):
        ctx.save_for_backward(axis)
        return raw_softmax(input, axis)

    @staticmethod
    def backward(ctx: Context, gy):
        axis, = ctx.saved_values
        y = ctx.outputs[0]()
        # gx = y * gy
        # sumdx = gx.sum(dim=axis, keepdims=True)
        # gx -= y * sumdx
        # return gx
        gx = fused_softmax_grad(y.data, gy.data, axis)
        return tensor(gx)

class SoftmaxCrossEntropy(Function):
    @staticmethod
    def forward(ctx:Context, logits, labels):
        loss = raw_softmax_crossentropy(logits, labels)
        # loss.data_sync(True)
        return loss

    @staticmethod
    def backward(ctx: Context, gy):
        logits, labels = ctx.inputs
        requires_grad = logits.requires_grad | labels.requires_grad
        grad = raw_softmax_crossentropy(logits.data, labels.data, True)
        grad = raw_mul(grad, gy.data)
        return tensor(grad, requires_grad=requires_grad), zeros_like(labels)

class SoftmaxCrossEntropyAscend(Function):
    @staticmethod
    def forward(ctx:Context, logits, labels):
        loss, grads = raw_softmax_crossentropy_ascend(logits, labels)
        ctx.save_for_backward(grads)
        return loss

    @staticmethod
    def backward(ctx:Context, gy):
        _, labels = ctx.inputs
        grads, = ctx.saved_tensors
        grad = grads * gy.reshape(-1, 1)
        return grad, zeros_like(labels)


class Linear(Function):
    @staticmethod
    def forward(ctx: Context, x, w, b):
        # y = raw_matmul(x, w, transpose_b=True)
        # if b is not None:
        #     y = raw_add(y, b)
        y = fused_linear(x, w, b)
        # y = raw_linear(x, w, b)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x, W, b = ctx.inputs
        # gb = None if b.data is None else sum_to(gy.data, b.shape)
        # gx = matmul(gy, W)
        # gW = matmul(x, gy, transpose_a=True)
        # return gx, gW.T, tensor(gb)
        gx, gw, gb = fused_linear_grad(x.data, W.data, b.data, gy.data)
        # gx, gw, _ = raw_linear_grad(x.data, W.data, gy.data)
        return tensor(gx), tensor(gw), tensor(gb)

class Conv2d(Function):
    @staticmethod
    def forward(ctx: Context, x, w, out_channel, kernel_size, pad_mode="valid", pad=0, stride=1, dilation=1, groups=1):
        ctx.save_for_backward(out_channel, kernel_size, pad_mode, pad, stride, dilation, groups)

        y = raw_conv2d(x, w, out_channel, kernel_size, pad_mode, pad, stride, dilation, groups)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x, w = ctx.inputs
        out_channel, kernel_size, pad_mode, pad, stride, dilation, groups = ctx.saved_tensors
        gx = conv2d_gx(gy, w, x.shape, out_channel, kernel_size, pad_mode, pad, pad, stride, dilation, groups)
        gw = conv2d_gw(gy, x, w.shape, out_channel, kernel_size, pad_mode, pad, pad, stride, dilation, groups)
        return gx, gw

def _conv2d(x, w, out_channel, kernel_size, pad_mode="valid", pad=0, stride=1, dilation=1, groups=1):
    return Conv2d.apply(x, w, out_channel=out_channel, kernel_size=kernel_size, pad_mode=pad_mode, pad=pad,
                        stride=stride, dilation=dilation, groups=groups)

class Conv2dGx(Function):
    @staticmethod
    def forward(ctx: Context, gy, w, x_shape, out_channel, kernel_size, pad_mode="valid", pad=0, pad_list=None,
               stride=1, dilation=1, groups=1):

        ctx.save_for_backward(out_channel, kernel_size, pad_mode, pad, pad_list, stride, dilation, groups)

        y = raw_conv2d_gx(gy, w, x_shape, out_channel, kernel_size, pad_mode, pad, pad_list, stride, dilation, groups)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x, w = ctx.inputs
        out_channel, kernel_size, pad_mode, pad, pad_list, stride, dilation, groups = ctx.saved_tensors
        gx = _conv2d(gy, w, out_channel, kernel_size, pad_mode, pad, stride, dilation, groups)
        gw = conv2d_gw(x, gy, w.shape, out_channel, kernel_size, pad_mode, pad, pad_list, stride, dilation, groups)
        return gx, gw

def conv2d_gx(gy, w, x_shape, out_channel, kernel_size, pad_mode="valid", pad=0, pad_list=None,
              stride=1, dilation=1, groups=1):
    return Conv2dGx.apply(gy, w, x_shape=x_shape, out_channel=out_channel, kernel_size=kernel_size, pad_mode=pad_mode,
                          pad=pad, pad_list=pad_list, stride=stride, dilation=dilation, groups=groups)

class Conv2dGw(Function):
    @staticmethod
    def forward(ctx: Context, gy, x, w_shape, out_channel, kernel_size, pad_mode="valid", pad=0, pad_list=None,
               stride=1, dilation=1, groups=1):

        ctx.save_for_backward(out_channel, kernel_size, pad_mode, pad, pad_list, stride, dilation, groups)

        y = raw_conv2d_gw(gy, x, w_shape, out_channel, kernel_size, pad_mode, pad, pad_list, stride, dilation, groups)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        dy, x = ctx.inputs
        out_channel, kernel_size, pad_mode, pad, pad_list, stride, dilation, groups = ctx.saved_tensors
        gx = _conv2d(x, gy, out_channel, kernel_size, pad_mode, pad, stride, dilation, groups)
        gw = conv2d_gx(dy, gy, x.shape, out_channel, kernel_size, pad_mode, pad, pad_list, stride, dilation, groups)
        return gx, gw

def conv2d_gw(gy, x, w_shape, out_channel, kernel_size, pad_mode="valid", pad=0, pad_list=None,
              stride=1, dilation=1, groups=1):
    return Conv2dGw.apply(gy, x, w_shape=w_shape, out_channel=out_channel, kernel_size=kernel_size, pad_mode=pad_mode.upper(),
                          pad=pad, pad_list=pad_list, stride=stride, dilation=dilation, groups=groups)


class BiasAdd(Function):
    @staticmethod
    def forward(ctx: Context, x0, x1):
        y = raw_bias_add(x0, x1)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        gb = raw_bias_add_grad(gy.data)
        return gy, tensor(gb)

def _bias_add(x, b):
    return BiasAdd.apply(x, b)

class Dropout(Function):
    @staticmethod
    def forward(ctx: Context, x, dropout):
        y, mask = raw_dropout(x, dropout)
        ctx.save_for_backward(mask, dropout)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        mask, dropout = ctx.saved_tensors
        gx = fused_dropout_grad(gy.data, mask, dropout)
        return tensor(gx)


class DropoutGrad(Function):
    @staticmethod
    def forward(ctx: Context, x, mask, dropout):
        gx = raw_dropout_grad(x, mask, dropout)
        ctx.save_for_backward(mask, dropout)
        return gx

    @staticmethod
    def backward(ctx: Context, gy):
        mask, dropout = ctx.saved_tensors
        mask = tensor(mask)
        gx = _dropout_grad(gy, mask, dropout)
        return gx, zeros_like(mask)

def _dropout_grad(x, mask, dropout):
    return DropoutGrad.apply(x, mask, dropout=dropout)


class MaxPool(Function):
    @staticmethod
    def forward(ctx: Context, x, kernel_size, strides=None, pads=0, dilation=(1, 1), ceil_mode=False, return_indices=False):
        out, indices = raw_maxpool(x, kernel_size, strides, pads, dilation, ceil_mode)
        ctx.save_for_backward(kernel_size, strides, pads, dilation, ceil_mode, indices)
        if return_indices:
            return out, indices
        return out

    @staticmethod
    def backward(ctx: Context, gy, indices=None):
        kernel_size, strides, pads, dilation, ceil_mode, indices = ctx.saved_tensors
        x, = ctx.inputs
        gx = _maxpool_grad(x, gy, tensor(indices), kernel_size, strides, pads, dilation, ceil_mode)
        return gx

def _maxpool(x, kernel_size, strides=None, pads=0, dilation=(1, 1), ceil_mode=False, return_indices=False):
    return MaxPool.apply(x, kernel_size=kernel_size, strides=strides, pads=pads,
                         dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)

class MaxPoolGrad(Function):
    @staticmethod
    def forward(ctx: Context, x, grad, argmax, kernel_size, strides=None, pads=0, dilation=(1, 1), ceil_mode=False):
        return raw_maxpool_grad(x, grad, argmax, kernel_size, strides, pads, dilation, ceil_mode)

def _maxpool_grad(x, grad, argmax, kernel_size, strides=None, pads=0, dilation=(1, 1), ceil_mode=False):
    return MaxPoolGrad.apply(x, grad, argmax, kernel_size=kernel_size, strides=strides, pads=pads,
                             dilation=dilation, ceil_mode=ceil_mode)

class NLLLoss(Function):
    @staticmethod
    def forward(ctx: Context, input, target, weight, ignore_index, reduction):
        out, total_weight = raw_nll_loss(input, target, weight, ignore_index, reduction)
        ctx.save_for_backward(ignore_index, reduction, total_weight)        
        return out

    @staticmethod
    def backward(ctx: Context, gy):
        input, target, weight = ctx.inputs
        ignore_index, reduction, total_weight = ctx.saved_tensors
        gx = raw_nll_loss_grad(input.data, gy.data, target.data, weight.data,
                               total_weight, ignore_index, reduction)
        return tensor(gx), zeros_like(target), zeros_like(weight)

class LayerNorm(Function):
    @staticmethod
    def forward(ctx: Context, input, weight, bias, begin_norm_axis=1, begin_params_axis=1, epsilon=1e-7):
        out, mean, var = raw_layer_norm(input, weight, bias, begin_norm_axis, begin_params_axis, epsilon)
        ctx.save_for_backward(mean, var, begin_norm_axis, begin_params_axis)
        return out

    @staticmethod
    def backward(ctx: Context, gy):
        input, weight, _ = ctx.inputs
        mean, var, begin_norm_axis, begin_params_axis = ctx.saved_tensors
        gx, gw, gb = raw_layer_norm_grad(input.data, gy.data, mean, var, weight.data,
                                         begin_norm_axis, begin_params_axis)
        return tensor(gx, requires_grad=gy.requires_grad), \
               tensor(gw, requires_grad=gy.requires_grad), \
               tensor(gb, requires_grad=gy.requires_grad)

class Unfold(Function):
    @staticmethod
    def forward(ctx: Context, input, kernel_size, dilation=1, padding=0, stride=1):
        return raw_unfold(input, kernel_size, stride, dilation, padding)

    @staticmethod
    def backward(ctx: Context, gy):
        pass


