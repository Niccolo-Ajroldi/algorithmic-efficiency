"""PyTorch implementation of ResNet.

Adapted from torchvision:
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py.
"""

import collections
from typing import Any, Callable, List, Optional, Type, Union

import torch
from torch import nn
from torch import Tensor

from algoperf import spec
from algoperf.init_utils import pytorch_default_init


def conv3x3(in_planes: int,
            out_planes: int,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1) -> nn.Conv2d:
  """3x3 convolution with padding."""
  return nn.Conv2d(
      in_planes,
      out_planes,
      kernel_size=3,
      stride=stride,
      padding=dilation,
      groups=groups,
      bias=False,
      dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
  """1x1 convolution."""
  return nn.Conv2d(
      in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
  """ResNet block."""
  expansion: int = 1

  def __init__(
      self,
      inplanes: int,
      planes: int,
      stride: int = 1,
      downsample: Optional[nn.Module] = None,
      groups: int = 1,
      base_width: int = 64,
      dilation: int = 1,
      norm_layer: Optional[Callable[..., nn.Module]] = None,
      act_fnc: nn.Module = nn.ReLU(inplace=True)
  ) -> None:
    super().__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    if groups != 1 or base_width != 64:
      raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
      raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    # Both self.conv1 and self.downsample layers downsample
    # the input when stride != 1.
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = norm_layer(planes)
    self.act_fnc = act_fnc
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x: spec.Tensor) -> spec.Tensor:
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.act_fnc(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.act_fnc(out)

    return out


class Bottleneck(nn.Module):
  """Bottleneck ResNet block."""
  expansion: int = 4

  def __init__(
      self,
      inplanes: int,
      planes: int,
      stride: int = 1,
      downsample: Optional[nn.Module] = None,
      groups: int = 1,
      base_width: int = 64,
      dilation: int = 1,
      norm_layer: Optional[Callable[..., nn.Module]] = None,
      act_fnc: nn.Module = nn.ReLU(inplace=True)
  ) -> None:
    super().__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    width = int(planes * (base_width / 64.)) * groups
    # Both self.conv2 and self.downsample layers downsample
    # the input when stride != 1.
    self.conv1 = conv1x1(inplanes, width)
    self.bn1 = norm_layer(width)
    self.conv2 = conv3x3(width, width, stride, groups, dilation)
    self.bn2 = norm_layer(width)
    self.conv3 = conv1x1(width, planes * self.expansion)
    self.bn3 = norm_layer(planes * self.expansion)
    self.act_fnc = act_fnc
    self.downsample = downsample
    self.stride = stride

  def forward(self, x: Tensor) -> Tensor:
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.act_fnc(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.act_fnc(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.act_fnc(out)

    return out


class ResNet(nn.Module):

  def __init__(self,
               block: Type[Union[BasicBlock, Bottleneck]],
               layers: List[int],
               num_classes: int = 1000,
               zero_init_residual: bool = True,
               groups: int = 1,
               width_per_group: int = 64,
               replace_stride_with_dilation: Optional[List[bool]] = None,
               norm_layer: Optional[Callable[..., nn.Module]] = None,
               act_fnc: nn.Module = nn.ReLU(inplace=True),
               bn_init_scale: float = 0.) -> None:
    super().__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    self._norm_layer = norm_layer

    self.inplanes = 64
    self.dilation = 1
    if replace_stride_with_dilation is None:
      # Each element in the tuple indicates if we should replace
      # the 2x2 stride with a dilated convolution instead.
      replace_stride_with_dilation = [False, False, False]
    if len(replace_stride_with_dilation) != 3:
      raise ValueError(
          'replace_stride_with_dilation should be None '
          f'or a 3-element tuple, got {replace_stride_with_dilation}')
    self.groups = groups
    self.base_width = width_per_group
    self.conv1 = nn.Conv2d(
        3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = norm_layer(self.inplanes)
    self.act_fnc = act_fnc
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, self.act_fnc, 64, layers[0])
    self.layer2 = self._make_layer(
        block,
        self.act_fnc,
        128,
        layers[1],
        stride=2,
        dilate=replace_stride_with_dilation[0])
    self.layer3 = self._make_layer(
        block,
        self.act_fnc,
        256,
        layers[2],
        stride=2,
        dilate=replace_stride_with_dilation[1])
    self.layer4 = self._make_layer(
        block,
        self.act_fnc,
        512,
        layers[3],
        stride=2,
        dilate=replace_stride_with_dilation[2])
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        pytorch_default_init(m)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    nn.init.normal_(self.fc.weight, std=1e-2)
    nn.init.constant_(self.fc.bias, 0.)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros,
    # and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to
    # https://arxiv.org/abs/1706.02677.
    if zero_init_residual:
      for m in self.modules():
        if isinstance(m, Bottleneck):
          nn.init.constant_(m.bn3.weight, bn_init_scale)
        elif isinstance(m, BasicBlock):
          nn.init.constant_(m.bn2.weight, bn_init_scale)

  def _make_layer(self,
                  block: Type[Union[BasicBlock, Bottleneck]],
                  act_fnc: nn.Module,
                  planes: int,
                  blocks: int,
                  stride: int = 1,
                  dilate: bool = False) -> nn.Sequential:
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
      self.dilation *= stride
      stride = 1
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = torch.nn.Sequential(
          collections.OrderedDict([
              ("conv", conv1x1(self.inplanes, planes * block.expansion,
                               stride)),
              ("bn", norm_layer(planes * block.expansion)),
          ]))

    layers = []
    layers.append(
        block(self.inplanes,
              planes,
              stride,
              downsample,
              self.groups,
              self.base_width,
              previous_dilation,
              norm_layer,
              act_fnc))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(
          block(
              self.inplanes,
              planes,
              groups=self.groups,
              base_width=self.base_width,
              dilation=self.dilation,
              norm_layer=norm_layer,
              act_fnc=act_fnc))

    return nn.Sequential(*layers)

  def forward(self, x: spec.Tensor) -> spec.Tensor:
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act_fnc(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x


def resnet18(**kwargs: Any) -> ResNet:
  return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet50(**kwargs: Any) -> ResNet:
  return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
