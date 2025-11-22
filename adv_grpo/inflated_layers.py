from functools import partial
from typing import Literal
from einops import rearrange
from torch import Tensor
from torch.nn import ConvTranspose2d, ConvTranspose3d

from flow_grpo.inflated_lib import (
    MemoryState,
    extend_head,
    inflate_bias,
    inflate_distribution_bias,
    inflate_distribution_weight,
    inflate_weight,
    modify_state_dict,
)
from flow_grpo.conv_gradfix import GradFixConv2d, GradFixConv3d

VERBOSE = False

_inflation_mode_t = (Literal["none", "flatten", "partial_flatten", "pad", "tile"],)
_direction_t = Literal["", "out", "in"]


class InflatedCausalConv3d(GradFixConv3d):
    """
    Note:
        To align the behavior of pretrained 2D models,
        if you compose a video clip from a single image by:
            - duplicating:      set shape_norm = True
            - padding zeros:    set shape_norm = False
        to avoid gaps in the beginning of training process.
    """

    def __init__(
        self, *args, inflation_mode: _inflation_mode_t, shape_norm: bool = True, **kwargs
    ):
        self.shape_norm = shape_norm
        self.inflation_mode = inflation_mode
        self.padding_bank = None
        super().__init__(*args, **kwargs)
        self.temporal_padding = self.padding[0]
        self.padding = (0, *self.padding[1:])  # Remove temporal pad to keep causal.

    def forward(self, input: Tensor, memory_state: MemoryState = MemoryState.DISABLED) -> Tensor:
        bank_size = self.stride[0] - self.kernel_size[0]
        padding_bank = (
            input[:, :, bank_size:].detach()
            if (bank_size != 0 and memory_state != MemoryState.DISABLED)
            else None
        )
        if (self.padding_bank is not None) and (memory_state == MemoryState.ACTIVE):
            input = extend_head(input, memory=self.padding_bank)
        else:
            input = extend_head(input, times=self.temporal_padding * 2)
        if memory_state != MemoryState.DISABLED and not self.training:
            self.padding_bank = padding_bank
        return super().forward(input)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.inflation_mode == "none":
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
        else:
            # NOTE: need to switch off strict
            super()._load_from_state_dict(
                modify_state_dict(
                    self,
                    state_dict,
                    prefix,
                    verbose=VERBOSE,
                    inflate_weight_fn=partial(inflate_weight, position="tail"),
                    inflate_bias_fn=partial(inflate_bias, position="tail"),
                ),
                prefix,
                local_metadata,
                False,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )


class InflatedDistributionCausalConv3d(GradFixConv3d):
    """
    Note:
        Direction:
            - out: this layer generates mean/std of some distribution;
            - in:  this layer takes tensors sampled from output of `out` layer as input.
    """

    def __init__(
        self,
        *args,
        direction: _direction_t,
        inflation_mode: _inflation_mode_t,
        shape_norm: bool = True,
        **kwargs,
    ):
        self.shape_norm = shape_norm
        self.inflation_mode = inflation_mode
        self.direction = direction
        self.padding_bank = None
        super().__init__(*args, **kwargs)
        self.temporal_padding = self.padding[0]
        self.padding = (0, *self.padding[1:])  # Remove temporal pad to keep causal.

    def forward(self, input: Tensor, memory_state: MemoryState = MemoryState.DISABLED) -> Tensor:
        bank_size = self.stride[0] - self.kernel_size[0]
        padding_bank = (
            input[:, :, bank_size:].detach()
            if (bank_size != 0 and memory_state != MemoryState.DISABLED)
            else None
        )
        if (self.padding_bank is not None) and (memory_state == MemoryState.ACTIVE):
            input = extend_head(input, memory=self.padding_bank)
        else:
            input = extend_head(input, times=self.temporal_padding * 2)
        if memory_state != MemoryState.DISABLED and not self.training:
            self.padding_bank = padding_bank
        return super().forward(input)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.inflation_mode == "none":
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
        else:
            super()._load_from_state_dict(
                modify_state_dict(
                    self,
                    state_dict,
                    prefix,
                    verbose=VERBOSE,
                    inflate_weight_fn=partial(
                        inflate_distribution_weight, direction=self.direction, position="tail"
                    ),
                    inflate_bias_fn=partial(
                        inflate_distribution_bias, direction=self.direction, position="tail"
                    ),
                ),
                prefix,
                local_metadata,
                False,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )


class InflatedConvTranspose3d(ConvTranspose3d):
    # Note: It's not a causal one.
    def __init__(
        self, *args, inflation_mode: _inflation_mode_t, shape_norm: bool = True, **kwargs
    ):
        self.shape_norm = shape_norm
        self.inflation_mode = inflation_mode
        super().__init__(*args, **kwargs)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.inflation_mode == "none":
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
        else:
            # NOTE: need to switch off strict
            super()._load_from_state_dict(
                modify_state_dict(
                    self,
                    state_dict,
                    prefix,
                    verbose=VERBOSE,
                    inflate_weight_fn=partial(inflate_weight, position="center"),
                    inflate_bias_fn=partial(inflate_bias, position="center"),
                ),
                prefix,
                local_metadata,
                False,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )


class FlattenedConvTranspose3d(ConvTranspose2d):
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        output = rearrange(input, "b c f h w -> (b f) c h w")
        output = super().forward(output)
        output = rearrange(output, "(b f) c h w -> b c f h w", f=input.size(2))
        return output


class FlattenedConv3d(GradFixConv2d):
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        output = rearrange(input, "b c f h w -> (b f) c h w")
        output = super().forward(output)
        output = rearrange(output, "(b f) c h w -> b c f h w", f=input.size(2))
        return output


def init_causal_conv3d(
    *args,
    inflation_mode: _inflation_mode_t,
    direction: _direction_t = "",
    partial_switch: bool = False,
    **kwargs,
):
    """
    Initialize a Causal-3D convolution layer.
    Parameters:
        inflation_mode: Listed as below. It's compatible with all the 3D-VAE checkpoints we have.
            - none: No inflation will be conducted.
                    The loading logic of state dict will fall back to default.
            - flatten:  It will produce a `fake` 3D layer,
                        which simply squeeze the axis of batch size and depth together,
                        and then conduct 2D convolution.
            - partial_flatten:
                - layers with `partial_switch` on:  using `none` mode.
                - layers with `partial_switch` off: using `flatten` mode.
            - pad / tile: Refer to the definition of `InflatedCausalConv3d`.
        direction:
            - empty string: Ordinary causal convolution layer.
            - out / in: Refer to the definition of `InflatedDistributionCausalConv3d`.
        partial_switch: Only works when `inflation_mode` is `partial_flatten`.
    """
    stride = kwargs.get("stride", args[3] if len(args) > 3 else None)
    padding = kwargs.get("padding", args[4] if len(args) > 4 else None)
    if "flatten" in inflation_mode:
        if (
            (
                (not stride)
                or isinstance(stride, int)
                or (isinstance(stride, list or tuple) and len(stride) < 3)
            )  # if the config of stride can be used for 2D conv
            and (
                (not padding)
                or isinstance(padding, int)
                or (isinstance(padding, list or tuple) and len(padding) < 3)
            )  # if the config of padding can be used for 2D conv
            and (("partial" not in inflation_mode) or (not partial_switch))
            # if it's fully-flatten mode, or with `partial_switch` off
        ):
            return FlattenedConv3d(*args, **kwargs)
        else:
            return InflatedCausalConv3d(*args, inflation_mode="none", **kwargs)
            # Force-override
    else:
        if direction:
            return InflatedDistributionCausalConv3d(
                *args, direction=direction, inflation_mode=inflation_mode, **kwargs
            )
        else:
            return InflatedCausalConv3d(*args, inflation_mode=inflation_mode, **kwargs)


def init_transposed_conv3d(
    *args, inflation_mode: _inflation_mode_t, partial_switch: bool = False, **kwargs
):
    stride = kwargs.get("stride", args[3] if len(args) > 3 else None)
    padding = kwargs.get("padding", args[4] if len(args) > 4 else None)
    if "flatten" in inflation_mode:
        if (
            (
                (not stride)
                or isinstance(stride, int)
                or (isinstance(stride, list or tuple) and len(stride) < 3)
            )
            and (
                (not padding)
                or isinstance(padding, int)
                or (isinstance(padding, list or tuple) and len(padding) < 3)
            )
            or (("partial" in inflation_mode) and not partial_switch)
        ):
            return FlattenedConvTranspose3d(*args, **kwargs)
        else:
            return InflatedConvTranspose3d(
                *args, inflation_mode="none", **kwargs
            )  # Force-override
    else:
        return InflatedConvTranspose3d(*args, inflation_mode=inflation_mode, **kwargs)