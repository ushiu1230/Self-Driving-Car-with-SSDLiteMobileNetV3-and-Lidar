import cv2
import numpy as np
import torch
import torchvision
import argparse
from PIL import Image
import torch.nn as nn
from functools import partial
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteHead , SSDLite320_MobileNet_V3_Large_Weights, SSDLiteClassificationHead, SSDLiteRegressionHead
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.ops.misc import ConvNormActivation, Conv2dNormActivation
from typing import Any, Callable, Dict, List, Optional, Union
from collections import OrderedDict
from functools import partial
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models import mobilenet
from torch import nn, Tensor
from torchvision.utils import _log_api_usage_once


def _extra_block(in_channels: int, out_channels: int, norm_layer: Callable[..., nn.Module]) -> nn.Sequential:
    activation = nn.ReLU6
    intermediate_channels = out_channels // 2
    return nn.Sequential(
        # 1x1 projection to half output channels
        Conv2dNormActivation(
            in_channels, intermediate_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=activation
        ),
        # 3x3 depthwise with stride 2 and padding 1
        Conv2dNormActivation(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=2,
            groups=intermediate_channels,
            norm_layer=norm_layer,
            activation_layer=activation,
        ),
        # 1x1 projetion to output channels
        Conv2dNormActivation(
            intermediate_channels, out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=activation
        ),
    )


def _normal_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.03)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)


class SSDLiteFeatureExtractorMobileNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        c4_pos: int,
        norm_layer: Callable[..., nn.Module],
        width_mult: float = 1.0,
        min_depth: int = 16,
    ):
        super().__init__()
        _log_api_usage_once(self)

        if backbone[c4_pos].use_res_connect:
            raise ValueError("backbone[c4_pos].use_res_connect should be False")

        self.features = nn.Sequential(
            # As described in section 6.3 of MobileNetV3 paper
            nn.Sequential(*backbone[:c4_pos], backbone[c4_pos].block[0]),  # from start until C4 expansion layer
            nn.Sequential(backbone[c4_pos].block[1:], *backbone[c4_pos + 1 :]),  # from C4 depthwise until end
        )

        get_depth = lambda d: max(min_depth, int(d * width_mult))  # noqa: E731
        extra = nn.ModuleList(
            [
                _extra_block(backbone[-1].out_channels, get_depth(512), norm_layer),
                _extra_block(get_depth(512), get_depth(256), norm_layer),
            ]
        )
        _normal_init(extra)

        self.extra = extra

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # Get feature maps from backbone and extra. Can't be refactored due to JIT limitations.
        output = []
        for block in self.features:
            x = block(x)
            output.append(x)

        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])

def _mobilenet_extractor(
    backbone: Union[mobilenet.MobileNetV2, mobilenet.MobileNetV3],
    trainable_layers: int,
    norm_layer: Callable[..., nn.Module],
):
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    num_stages = len(stage_indices)

    # find the index of the layer from which we wont freeze
    if not 0 <= trainable_layers <= num_stages:
        raise ValueError("trainable_layers should be in the range [0, {num_stages}], instead got {trainable_layers}")
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    return SSDLiteFeatureExtractorMobileNet(backbone, stage_indices[-2], norm_layer)

weights: Optional[SSDLite320_MobileNet_V3_Large_Weights] = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
weights_backbone = None
progress: bool = True
weights_backbone: None
trainable_backbone_layers: Optional[int] = None

weights = SSDLite320_MobileNet_V3_Large_Weights.verify(weights)
weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)
reduce_tail = weights_backbone is None
norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

trainable_backbone_layers = _validate_trainable_layers(weights is not None or weights_backbone is not None, trainable_backbone_layers, 6, 6)

backbone = mobilenet_v3_large(weights=weights_backbone, progress=progress, norm_layer=norm_layer, reduced_tail=reduce_tail)

_normal_init(backbone)
backbone = _mobilenet_extractor(backbone, trainable_backbone_layers, norm_layer)

def Model4(device):
  model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights= SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
  model.backbone = backbone
  in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
  model.anchor_generator = DefaultBoxGenerator([[1.5],[2,3],[2,3],[2]], min_ratio=0.2, max_ratio=0.9)
  num_anchors = model.anchor_generator.num_anchors_per_location()
  aspect_ratios = model.anchor_generator.aspect_ratios
  norm_layer  = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
  model.head.regression_head = SSDLiteRegressionHead(in_channels, num_anchors, norm_layer)
  model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, 2, norm_layer)
  checkpoint = torch.load("models/M4.pth", map_location = device)
  model.load_state_dict(checkpoint['model'])
  model.eval()
  return model

def Model3(device):
  model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights= SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
  model.backbone = backbone
  in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
  model.anchor_generator = DefaultBoxGenerator([[1.5],[2,3],[2,3],[2]], min_ratio=0.2, max_ratio=0.9)
  num_anchors = model.anchor_generator.num_anchors_per_location()
  aspect_ratios = model.anchor_generator.aspect_ratios
  norm_layer  = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
  model.head.regression_head = SSDLiteRegressionHead(in_channels, num_anchors, norm_layer)
  model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, 8, norm_layer)
  checkpoint = torch.load("models/700.pth", map_location = device)
  model.load_state_dict(checkpoint['model'])
  model.eval()
  return model

def Model1(device):
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer  = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, 2, norm_layer)
    checkpoint = torch.load("models/M1.pth", map_location = device)
    model.load_state_dict(checkpoint['model'])    
    model.eval()
    return model

def Model0(device):
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights= SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    model.eval()
    return model