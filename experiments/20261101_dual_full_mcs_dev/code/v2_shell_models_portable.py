"""用于读取现有 v2 checkpoint 的最小 SHELL 模型定义。

结构与历史训练时的 Res1dNet31、M5 和 dual_v6 checkpoint 键严格对应；
本文件只服务只读推理，不改变 checkpoint。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_layer(layer: nn.Module) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if getattr(layer, "bias", None) is not None:
        layer.bias.data.fill_(0.0)


def _init_bn(layer: nn.Module) -> None:
    layer.bias.data.fill_(0.0)
    layer.weight.data.fill_(1.0)


def _conv3(in_planes: int, out_planes: int, dilation: int) -> nn.Conv1d:
    return nn.Conv1d(in_planes, out_planes, 3, stride=1, padding=dilation, bias=False, dilation=dilation)


def _conv1(in_planes: int, out_planes: int) -> nn.Conv1d:
    return nn.Conv1d(in_planes, out_planes, 1, stride=1, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None, groups: int = 1,
                 base_width: int = 64, dilation: int = 1, norm_layer=None):
        super().__init__()
        norm_layer = nn.BatchNorm1d if norm_layer is None else norm_layer
        if groups != 1 or base_width != 64 or dilation > 1:
            raise ValueError("Res1dNet31 基础块参数不受支持")
        self.stride = stride
        self.conv1 = _conv3(inplanes, planes, 1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3(planes, planes, 2)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        _init_layer(self.conv1)
        _init_bn(self.bn1)
        _init_layer(self.conv2)
        _init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x
        out = F.max_pool1d(x, kernel_size=self.stride) if self.stride != 1 else x
        out = self.relu(self.bn1(self.conv1(out)))
        out = F.dropout(out, p=0.1, training=self.training)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(out + identity)


class _ResNet1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 3, 2)
        self.layer3 = self._make_layer(256, 4, 2)
        self.layer4 = self._make_layer(512, 3, 2)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            if stride == 1:
                downsample = nn.Sequential(_conv1(self.inplanes, planes), nn.BatchNorm1d(planes))
                _init_layer(downsample[0])
                _init_bn(downsample[1])
            else:
                downsample = nn.Sequential(nn.AvgPool1d(stride), _conv1(self.inplanes, planes), nn.BatchNorm1d(planes))
                _init_layer(downsample[1])
                _init_bn(downsample[2])
        layers = [_BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers.extend(_BasicBlock(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class Res1dNet31(nn.Module):
    def __init__(self, in_channel: int, classes_num: int):
        super().__init__()
        self.conv0 = nn.Conv1d(in_channel, 64, 1, bias=False)
        self.bn0 = nn.BatchNorm1d(64)
        self.resnet = _ResNet1d()
        self.fc_audioset = nn.Linear(512, classes_num)
        _init_layer(self.conv0)
        _init_bn(self.bn0)
        _init_layer(self.fc_audioset)

    def forward(self, x):
        x = self.resnet(self.bn0(self.conv0(x)))
        return self.fc_audioset(torch.max(x, dim=2)[0] + torch.mean(x, dim=2))


class M5(nn.Module):
    def __init__(self, n_input: int = 1, n_output: int = 35, stride: int = 1,
                 n_channel: int = 128, kernel_size_first: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size_first, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, 3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, 3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, 3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = F.avg_pool1d(x, x.shape[-1]).permute(0, 2, 1)
        return self.fc1(x)


class DualV6(nn.Module):
    def __init__(self, in_channels: int = 16, num_classes: int = 13):
        super().__init__()
        self.num_classes = num_classes
        self.legacy = Res1dNet31(in_channels, 2 * num_classes)

    def forward(self, x):
        return self.legacy(x).reshape(x.shape[0], 2, self.num_classes)


def build_model(task: str, model_kind: str) -> nn.Module:
    if model_kind == "spectrum":
        return M5(n_input=16, n_output=24)
    if task == "single":
        return Res1dNet31(in_channel=16, classes_num=13)
    return DualV6(in_channels=16, num_classes=13)
