"""
Custom replacement for `torch.nn.functional.convNd` and `torch.nn.functional.conv_transposeNd`
that supports arbitrarily high order gradients with zero performance penalty.
Modified from https://github.com/NVlabs/stylegan3/blob/main/torch_utils/ops/conv2d_gradfix.py
"""

import contextlib
import warnings
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv2d, Conv3d

# pylint: disable=redefined-builtin
# pylint: disable=arguments-differ
# pylint: disable=protected-access

# ----------------------------------------------------------------------------

enabled = False  # Enable the custom op by setting this to true.
weight_gradients_disabled = (
    False  # Forcefully disable computation of gradients with respect to the weights.
)


@contextlib.contextmanager
def no_weight_gradients():
    global weight_gradients_disabled
    old = weight_gradients_disabled
    weight_gradients_disabled = True
    yield
    weight_gradients_disabled = old


# ----------------------------------------------------------------------------
class GradFixConv2d(Conv2d):
    def __init__(self, *args, use_gradfix: bool = False, **kwargs):
        self.use_gradfix = use_gradfix
        super().__init__(*args, **kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        conv_fn = F.conv2d if not self.use_gradfix else convNd
        if self.padding_mode != "zeros":
            return conv_fn(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight,
                bias,
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )
        return conv_fn(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(
        self, input: Tensor, weight: Optional[Tensor] = None, bias: Optional[Tensor] = None
    ) -> Tensor:
        weight = self.weight if weight is None else weight
        bias = self.bias if bias is None else bias
        return self._conv_forward(input, weight, bias)


class GradFixConv3d(Conv3d):
    def __init__(self, *args, use_gradfix: bool = False, **kwargs):
        self.use_gradfix = use_gradfix
        super().__init__(*args, **kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        conv_fn = F.conv3d if not self.use_gradfix else convNd
        if self.padding_mode != "zeros":
            return conv_fn(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight,
                bias,
                self.stride,
                (0, 0, 0),
                self.dilation,
                self.groups,
            )
        return conv_fn(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(
        self, input: Tensor, weight: Optional[Tensor] = None, bias: Optional[Tensor] = None
    ) -> Tensor:
        weight = self.weight if weight is None else weight
        bias = self.bias if bias is None else bias
        return self._conv_forward(input, weight, bias)


# ----------------------------------------------------------------------------


def convNd(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    N = weight.ndim - 2
    if _should_use_custom_op(input):
        return _conv_gradfix(
            transpose=False,
            weight_shape=weight.shape,
            stride=stride,
            padding=padding,
            output_padding=0,
            dilation=dilation,
            groups=groups,
        ).apply(input, weight, bias)
    return getattr(torch.nn.functional, f"conv{N}d")(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def conv_transposeNd(
    input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1
):
    N = weight.ndim - 2
    if _should_use_custom_op(input):
        return _conv_gradfix(
            transpose=True,
            weight_shape=weight.shape,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        ).apply(input, weight, bias)
    return getattr(torch.nn.functional, f"conv_transpose{N}d")(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )


# ----------------------------------------------------------------------------


def _should_use_custom_op(input):
    assert isinstance(input, torch.Tensor)
    if (not enabled) or (not torch.backends.cudnn.enabled):
        return False
    if input.device.type != "cuda":
        return False
    if any(torch.__version__.startswith(x) for x in ["1.7.", "1.8.", "1.9"]):
        return True
    if torch.__version__.startswith("2"):
        return True
    warnings.warn(
        f"conv2d_gradfix not supported on PyTorch {torch.__version__}. "
        f"Falling back to torch.nn.functional.conv2d()."
    )
    return False


def _tuple_of_ints(xs, ndim):
    xs = tuple(xs) if isinstance(xs, (tuple, list)) else (xs,) * ndim
    assert len(xs) == ndim
    assert all(isinstance(x, int) for x in xs)
    return xs


# ----------------------------------------------------------------------------

_conv_gradfix_cache = dict()


def _conv_gradfix(transpose, weight_shape, stride, padding, output_padding, dilation, groups):
    ndim = len(weight_shape) - 2
    # Parse arguments.
    weight_shape = tuple(weight_shape)
    stride = _tuple_of_ints(stride, ndim)
    padding = _tuple_of_ints(padding, ndim)
    output_padding = _tuple_of_ints(output_padding, ndim)
    dilation = _tuple_of_ints(dilation, ndim)

    # Lookup from cache.
    key = (transpose, weight_shape, stride, padding, output_padding, dilation, groups)
    if key in _conv_gradfix_cache:
        return _conv_gradfix_cache[key]

    # Validate arguments.
    assert groups >= 1
    assert all(stride[i] >= 1 for i in range(ndim))
    assert all(padding[i] >= 0 for i in range(ndim))
    assert all(dilation[i] >= 0 for i in range(ndim))
    if not transpose:
        assert all(output_padding[i] == 0 for i in range(ndim))
    else:  # transpose
        assert all(0 <= output_padding[i] < max(stride[i], dilation[i]) for i in range(ndim))

    # Helpers.
    common_kwargs = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)

    def calc_output_padding(input_shape, output_shape):
        if transpose:
            return [
                0,
            ] * ndim
        return [
            input_shape[i + 2]
            - (output_shape[i + 2] - 1) * stride[i]
            - (1 - 2 * padding[i])
            - dilation[i] * (weight_shape[i + 2] - 1)
            for i in range(ndim)
        ]

    # Forward & backward.
    class ConvNd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            """
            input size: [B, C, ...]
            weight size:
                -> Conv:        [C_out, C_in // groups, ...]
                -> Transpose:   [C_in, C_out // groups, ...]
            """
            assert weight.shape == weight_shape
            ctx.save_for_backward(input, weight)

            # General case => cuDNN.
            if transpose:
                return getattr(torch.nn.functional, f"conv_transpose{ndim}d")(
                    input=input,
                    weight=weight.to(input.dtype),
                    bias=bias,
                    output_padding=output_padding,
                    **common_kwargs,
                )
            return getattr(torch.nn.functional, f"conv{ndim}d")(
                input=input, weight=weight.to(input.dtype), bias=bias, **common_kwargs
            )

        @staticmethod
        def backward(ctx, grad_output):
            input, weight = ctx.saved_tensors
            grad_input = None
            grad_weight = None
            grad_bias = None

            if ctx.needs_input_grad[0]:  # Input
                p = calc_output_padding(input_shape=input.shape, output_shape=grad_output.shape)
                op = _conv_gradfix(
                    transpose=(not transpose),
                    weight_shape=weight_shape,
                    output_padding=p,
                    **common_kwargs,
                )
                grad_input = op.apply(grad_output, weight, None)
                assert grad_input.shape == input.shape

            if ctx.needs_input_grad[1] and not weight_gradients_disabled:  # Weight
                grad_weight = ConvNdGradWeight.apply(grad_output, input)
                assert grad_weight.shape == weight_shape

            if ctx.needs_input_grad[2]:  # Bias
                grad_bias = grad_output.transpose(0, 1).flatten(1).sum(1)

            return grad_input, grad_weight, grad_bias

    # Gradient with respect to the weights.
    class ConvNdGradWeight(torch.autograd.Function):
        @staticmethod
        def forward(ctx, grad_output, input):
            flags = [
                torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic,
                torch.backends.cudnn.allow_tf32,
            ]
            if torch.__version__.startswith("1"):
                op = torch._C._jit_get_operation(
                    "aten::cudnn_convolution_backward_weight"
                    if not transpose
                    else "aten::cudnn_convolution_transpose_backward_weight"
                )
                grad_weight = op(
                    weight_shape,
                    grad_output,
                    input.to(grad_output.dtype),
                    padding,
                    stride,
                    dilation,
                    groups,
                    *flags,
                )
            elif torch.__version__.startswith("2"):
                # https://github.com/pytorch/pytorch/issues/74437
                op, _ = torch._C._jit_get_operation("aten::convolution_backward")
                dummy_weight = torch.tensor(
                    0.0, dtype=grad_output.dtype, device=input.device
                ).expand(weight_shape)
                grad_weight = op(
                    grad_output,
                    input.to(grad_output.dtype),
                    dummy_weight,
                    None,
                    stride,
                    padding,
                    dilation,
                    transpose,
                    (0,) * ndim,
                    groups,
                    [False, True, False],
                )[1]
            else:
                raise NotImplementedError
            assert grad_weight.shape == weight_shape
            ctx.save_for_backward(grad_output, input)
            return grad_weight

        @staticmethod
        def backward(ctx, grad2_grad_weight):
            grad_output, input = ctx.saved_tensors
            grad2_grad_output = None
            grad2_input = None

            if ctx.needs_input_grad[0]:  # Grad of Weight
                grad2_grad_output = ConvNd.apply(input, grad2_grad_weight, None)
                assert grad2_grad_output.shape == grad_output.shape

            if ctx.needs_input_grad[1]:  # Input
                p = calc_output_padding(input_shape=input.shape, output_shape=grad_output.shape)
                op = _conv_gradfix(
                    transpose=(not transpose),
                    weight_shape=weight_shape,
                    output_padding=p,
                    **common_kwargs,
                )
                grad2_input = op.apply(grad_output, grad2_grad_weight, None)
                assert grad2_input.shape == input.shape

            return grad2_grad_output, grad2_input

    _conv_gradfix_cache[key] = ConvNd
    return ConvNd


# ----------------------------------------------------------------------------
