import math
from enum import Enum
from typing import Optional
import numpy as np
import torch
from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.normalization import RMSNorm
from einops import rearrange
from torch import Tensor, nn

# from common.logger import get_logger

# logger = get_logger(__name__)


class MemoryState(Enum):
    """
    State[Disabled]:        No memory bank will be enabled.
    State[Initializing]:    The model is handling the first clip,
                            need to reset / initialize the memory bank.
    State[Active]:          There has been some data in the memory bank.
    """

    DISABLED = 0
    INITIALIZING = 1
    ACTIVE = 2


def norm_wrapper(
    norm_layer: nn.Module,
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    keep_causal: bool = False,
) -> torch.Tensor:
    if isinstance(norm_layer, (nn.LayerNorm, RMSNorm)):
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b h w c")
            x = norm_layer(x)
            x = rearrange(x, "b h w c -> b c h w")
            return x
        if x.ndim == 5:
            x = rearrange(x, "b c t h w -> b t h w c")
            x = norm_layer(x)
            x = rearrange(x, "b t h w c -> b c t h w")
            return x
    if isinstance(norm_layer, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
        if x.ndim <= 4 or (not keep_causal and not isinstance(norm_layer, nn.BatchNorm2d)):
            return norm_layer(x)
        if x.ndim == 5:
            t = x.size(2)
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = norm_layer(x)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
            return x
    if isinstance(norm_layer, SpatialNorm):
        t = -1
        if x.ndim == 5:
            t = x.size(2)
            x = rearrange(x, "b c t h w -> (b t) c h w")
        if y.ndim == 5:
            y = rearrange(y, "b c t h w -> (b t) c h w")
        if x.ndim != 4 or y.ndim != 4:
            raise NotImplementedError
        x = norm_layer(x, y)
        if t != -1:
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x
    raise NotImplementedError


def remove_head(tensor: Tensor, times: int = 1) -> Tensor:
    """
    Remove duplicated first frame features in the up-sampling process.
    """
    if times == 0:
        return tensor
    return torch.cat(tensors=(tensor[:, :, :1], tensor[:, :, times + 1 :]), dim=2)


def extend_head(
    tensor: Tensor, times: Optional[int] = 2, memory: Optional[Tensor] = None
) -> Tensor:
    """
    When memory is None:
        - Duplicate first frame features in the down-sampling process.
    When memory is not None:
        - Concatenate memory features with the input features to keep temporal consistency.
    """
    if times == 0:
        return tensor
    if memory is not None:
        return torch.cat((memory.to(tensor), tensor), dim=2)
    else:
        tile_repeat = np.ones(tensor.ndim).astype(int)
        tile_repeat[2] = times
        return torch.cat(tensors=(torch.tile(tensor[:, :, :1], list(tile_repeat)), tensor), dim=2)


def fill_weight_in_depth(weight: torch.Tensor, source: torch.Tensor, position: str):
    """
    Inflate a 2D convolution weight matrix to a 3D one by padding zeros in the channel of depth.
    Parameters:
         weight: The weight parameters of 3D conv kernel to be initialized.
         source: The weight parameters of 2D conv kernel to be inflated.
         position: Where to insert the 2D weights, can be chosen from
            - tail: Pad zeros in the front of the 2D kernel. Used for casual inflation.
            - center: Pad zeros around the 2D kernel. Used for normal inflation.
    """
    assert position in ["tail", "center"], "Unsupported fill-in position for weight inflation."
    depth = weight.size(2)
    weight.fill_(0.0)
    if position == "center":
        if depth % 2 == 1:
            weight[:, :, depth // 2].copy_(source.squeeze(2))
        else:
            weight[:, :, depth // 2].copy_(source.squeeze(2) / 2.0)
            weight[:, :, depth // 2 - 1].copy_(source.squeeze(2) / 2.0)
    else:
        if depth % 2 == 1:
            weight[:, :, -1].copy_(source.squeeze(2))
        else:
            weight[:, :, -1].copy_(source.squeeze(2) / 2.0)
            weight[:, :, -2].copy_(source.squeeze(2) / 2.0)
    return weight


def inflate_weight(
    weight_2d: torch.Tensor,
    weight_3d: torch.Tensor,
    shape_norm: bool,
    name: str,
    inflation_mode: str,
    position: str,
    verbose: bool = True,
):
    """
    Inflate a 2D convolution weight matrix to a 3D one.
    Parameters:
        weight_2d:      The weight matrix of 2D conv to be inflated.
        weight_3d:      The weight matrix of 3D conv to be initialized.
        inflation_mode: the mode of inflation
            - pad: pad zeros around 2D kernel.
            - tile: tile 2D kernel along the depth axis.

        shape_norm:     Whether to scale the parameters of 2D kernel so that the untrained
                        inflated model behaves exactly the same as the original 2D model
                        in the reconstruction of image and video. recommend to switch it on.

        name:           The name of inflated module. Only be used in logging.
        position:       Refer to the doc of `fill_weight_in_depth`.
                        Only works when `inflation_mode` is `pad`.
        verbose:        Whether to log information about inflation.
    """
    assert inflation_mode in ["pad", "tile"]
    depth = weight_3d.size(2)
    tgt_out, tgt_in = weight_3d.size()[:2]
    src_out, src_in = weight_2d.size()[:2]
    assert (tgt_out % src_out == 0) and (tgt_in % src_in == 0)
    out_fan, in_fan = tgt_out // src_out, tgt_in // src_in
    depth_factor = 1 if inflation_mode == "pad" else depth
    factor = (depth_factor * math.sqrt(out_fan) * math.sqrt(in_fan)) if shape_norm else 1
    with torch.no_grad():
        channel_inflation = weight_2d.unsqueeze(2).repeat(out_fan, in_fan, 1, 1, 1) / factor
        if inflation_mode == "tile":
            weight_3d.copy_(channel_inflation.repeat(1, 1, depth, 1, 1))
        else:
            weight_3d = fill_weight_in_depth(weight_3d, channel_inflation, position)
        if verbose:
            print(
                f"*** {name}weight {weight_2d.size()} is inflated to {weight_3d.size()} ***"
            )
        return weight_3d


def inflate_bias(
    bias_2d: torch.Tensor,
    bias_3d: torch.Tensor,
    shape_norm: bool,
    name: str,
    inflation_mode: str,
    position: str,
    verbose: bool = True,
):
    """
    Inflate a 2D convolution bias tensor to a 3D one
    Parameters:
        bias_2d:        The bias tensor of 2D conv to be inflated.
        bias_3d:        The bias tensor of 3D conv to be initialized.
        shape_norm:     Refer to `inflate_weight` function.
        name:           The name of inflated module. Only be used in logging.
        inflation_mode: Placeholder to align `inflate_weight`.
        position:       Placeholder to align `inflate_weight`.
        verbose:        Whether to log information about inflation.
    """
    tgt_ch, src_ch = bias_3d.size(0), bias_2d.size(0)
    assert tgt_ch % src_ch == 0
    fan = tgt_ch // src_ch
    factor = math.sqrt(fan) if shape_norm else 1
    with torch.no_grad():
        bias_3d.copy_(bias_2d.repeat(fan) / factor)
        if (tgt_ch != src_ch) and verbose:
            print(f"*** {name}bias {bias_2d.size()} is inflated to {bias_3d.size()} ***")
        return bias_3d


def inflate_distribution_weight(
    weight_2d: torch.Tensor,
    weight_3d: torch.Tensor,
    shape_norm: bool,
    name: str,
    direction: str,
    inflation_mode: str,
    position: str,
    verbose: bool = True,
):
    """
    Inflate a 2D convolution weight matrix to a 3D one.
    Note:   Different from `inflate_weight`,
            it's designed for `quant_conv` or `post_quant_conv` layers.
            i.e.,   a convolution layer used to produce `mean` and `std` of some distribution,
                    or its subsequent layer.
    Parameters: Refer to `inflate_weight`.
        direction:
            - out:  this layer generates `mean` and `std` of some distribution.
            - in:   this layer takes tensors sampled from output of `out` layer as input.
    """
    assert inflation_mode in ["pad", "tile"]
    depth = weight_3d.size(2)
    tgt_out, tgt_in = weight_3d.size()[:2]
    src_out, src_in = weight_2d.size()[:2]
    assert (tgt_out % src_out == 0) and (tgt_in % src_in == 0)
    out_fan, in_fan = tgt_out // src_out, tgt_in // src_in
    depth_factor = 1 if inflation_mode == "pad" else depth
    if direction == "out":
        factor = (depth_factor * math.sqrt(in_fan)) if shape_norm else 1
        with torch.no_grad():
            in_inflation = weight_2d.unsqueeze(2).repeat(1, in_fan, 1, 1, 1) / factor
            # [src_out, src_in, k_h, k_w] -> [src_out, tgt_in, 1, k_h, k_w]
            out_mean_weight, out_std_weight = torch.chunk(in_inflation, 2, dim=0)
            mean_slice = slice(src_out // 2)
            std_slice = slice(tgt_out // 2, tgt_out // 2 + src_out // 2)
            if inflation_mode == "tile":
                weight_3d[mean_slice] = out_mean_weight
                weight_3d[std_slice] = out_std_weight
                # Other part will be randomly initialized.
            else:
                weight_3d[mean_slice] = fill_weight_in_depth(
                    weight_3d[mean_slice], out_mean_weight, position
                )
                weight_3d[std_slice] = fill_weight_in_depth(
                    weight_3d[std_slice], out_std_weight, position
                )
                # Other part will be randomly initialized.
    elif direction == "in":
        factor = (depth_factor * math.sqrt(out_fan)) if shape_norm else 1
        with torch.no_grad():
            out_inflation = weight_2d.unsqueeze(2).repeat(out_fan, 1, 1, 1, 1) / factor
            # [src_out, src_in, k_h, k_w] -> [tgt_out, src_in, 1, k_h, k_w]
            if inflation_mode == "tile":
                weight_3d[:, :src_in] = out_inflation
            else:
                weight_3d[:, :src_in] = fill_weight_in_depth(
                    weight_3d[:, :src_in], out_inflation, position
                )
            weight_3d[:, src_in:].fill_(0.0)
    else:
        raise NotImplementedError
    if verbose:
        print(
            f"*** [Distribution] {name}weight {weight_2d.size()} "
            f"is inflated to {weight_3d.size()} ***"
        )
    return weight_3d


def inflate_distribution_bias(
    bias_2d: torch.Tensor,
    bias_3d: torch.Tensor,
    shape_norm: bool,
    name: str,
    direction: str,
    inflation_mode: str,
    position: str,
    verbose: bool = True,
):
    """
    The combination of `inflate_distribution_weight` and `inflate_bias`.
    """
    tgt_ch, src_ch = bias_3d.size(0), bias_2d.size(0)
    assert tgt_ch % src_ch == 0
    if direction == "out":
        with torch.no_grad():
            out_mean_bias, out_std_bias = torch.chunk(bias_2d, 2, dim=0)
            bias_3d[: src_ch // 2] = out_mean_bias
            bias_3d[tgt_ch // 2 : tgt_ch // 2 + src_ch // 2] = out_std_bias
    elif direction == "in":
        with torch.no_grad():
            bias_3d[:src_ch] = bias_2d
            bias_3d[src_ch:].fill_(0.0)
    else:
        raise NotImplementedError
    if verbose:
        print(
            f"*** [Distribution] {name}bias {bias_2d.size()} is inflated to {bias_3d.size()} ***"
        )
    return bias_3d


def modify_state_dict(
    layer, state_dict, prefix, inflate_weight_fn, inflate_bias_fn, verbose=False
):
    """
    the main function to inflated 2D parameters to 3D.
    """
    weight_name = prefix + "weight"
    bias_name = prefix + "bias"
    if weight_name in state_dict:
        weight_2d = state_dict[weight_name]
        if (
            weight_2d.dim() == 4
        ):  # Assuming the 2D weights are 4D tensors (out_channels, in_channels, h, w)
            weight_3d = inflate_weight_fn(
                weight_2d=weight_2d,
                weight_3d=layer.weight,
                shape_norm=layer.shape_norm,
                name=prefix,
                verbose=verbose,
                inflation_mode=layer.inflation_mode,
            )
            state_dict[weight_name] = weight_3d
        else:
            return state_dict
            # It's a 3d state dict, should not do inflation on both bias and weight.
    if bias_name in state_dict:
        bias_2d = state_dict[bias_name]
        if bias_2d.dim() == 1:  # Assuming the 2D biases are 1D tensors (out_channels,)
            bias_3d = inflate_bias_fn(
                bias_2d=bias_2d,
                bias_3d=layer.bias,
                shape_norm=layer.shape_norm,
                name=prefix,
                verbose=verbose,
                inflation_mode=layer.inflation_mode,
            )
            state_dict[bias_name] = bias_3d
    return state_dict