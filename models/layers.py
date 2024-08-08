import torch.nn.functional as F
import torch.nn as nn
import math
import torch
from torch import nn


def instance_std(x, eps=1e-5):
    var = torch.var(x, dim=(2, 3), keepdim=True).expand_as(x)
    if torch.isnan(var).any():
        var = torch.zeros(var.shape)
    return torch.sqrt(var + eps)

def group_std(x, groups=32, eps=1e-5):
    N, C, H, W = x.size()
    x = torch.reshape(x, (N, groups, C // groups, H, W))
    var = torch.var(x, dim=(2, 3, 4), keepdim=True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), (N, C, H, W))

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class EvoNorm2D(nn.Module):
    def __init__(
        self, 
        input, 
        groups=32, 
        non_linear=True, 
        version="S0", 
        affine=True, 
        momentum=0.9, 
        eps=1e-5, 
        training=True
    ):
        super(EvoNorm2D, self).__init__()
        self.non_linear = non_linear
        self.version = version
        self.training = training
        self.momentum = momentum
        self.groups = groups
        self.eps = eps
        if self.version not in ["B0", "S0"]:
            raise ValueError("Invalid EvoNorm version")
        self.insize = input
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, self.insize, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, self.insize, 1, 1))
            if self.non_linear:
                self.v = nn.Parameter(torch.ones(1, self.insize, 1, 1))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
            self.register_buffer("v", None)
        self.register_buffer("running_var", torch.ones(1, self.insize, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(x.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.version == "S0":
            if self.non_linear:
                num = x * torch.sigmoid(self.v * x)
                return num / group_std(x, groups=self.groups,eps=self.eps) * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
        if self.version == "B0":
            if self.training:
                var = torch.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var)
            else:
                var = self.running_var

            if self.non_linear:
                den = torch.max((var + self.eps).sqrt(), self.v * x + instance_std(x, eps=self.eps))
                return x / den * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta


class BasicConv(nn.Module):
    def __init__(self, 
        in_planes, 
        out_planes, 
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1, 
        bias=False
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, 
        gate_channels, 
        reduction_ratio=16, 
        pool_types=['avg', 'max']
    ):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
