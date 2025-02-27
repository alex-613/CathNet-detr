# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
        # Freeze some of the parameters, only trainable layers are 2 3 and 4
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            # we extract features from layer 4 and put it into a dictionary under the index of '0'
            return_layers = {'layer4': "0"}
        # Finally take resnet 50, call intermediate layer getter, which only grabs layer 4 from the intermediate layer getter
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        # Pass tensors through the body, which will just fetch features of layer 4 of resnet 50. xs will have zero key, the 4th layer. With batch, 2048, feature dims.
        # Basically the extracted features
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            # Grab the features from xs. Grab the mask
            m = tensor_list.mask
            assert m is not None
            # Downsample the mask such that it isnt of the same spatial resolution of the original image, but it is of the same spatial res of resnet 50.
            # Eg the orig m.shape is [2, 608, 911]
            # the interpolated shape is [2, 19, 29], which is of the same res of output of resnet 50
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            # Finally, store the image features and masks in a nested vector and return it in the out dictionary
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        # Backbone is the resnet 50 and grab the class
        # Grab regular resnet without dilated convolution, which is a fancy way of downsampling, not max pooling
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        # self[0] is just the backbone ResNet50
        # self[1] is just the positional embedding with sine
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        # Now we want to calculate the positional embeddings
        pos = []
        # Grab image features and down samples mask
        for name, x in xs.items():
            out.append(x)
            # position encoding
            # Give it the x and it does the magic. Treating it as a blackbox for now.
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    # Add positional encoding
    position_embedding = build_position_encoding(args)
    # If learning rate is greater than 0, make the backbone trainable
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    # Now proceed to making the backbone, you will need to modify this if you would like to conv temporal information too.
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    # Joiner joins the resnet 50 with the positional encoder
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

