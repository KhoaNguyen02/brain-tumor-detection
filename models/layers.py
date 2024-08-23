import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn import init
from torch.nn.parameter import Parameter


def instance_std(x, eps=1e-5):
    N, C, H, W = x.size()
    x1 = x.reshape(N*C, -1)
    var = x1.var(dim=-1, keepdim=True)+eps
    return var.sqrt().reshape(N, C, 1, 1)


def group_std(x, groups, eps=1e-5):
    N, C, H, W = x.size()
    x1 = x.reshape(N, groups, -1)
    var = (x1.var(dim=-1, keepdim=True)+eps).reshape(N, groups, -1)
    return (x1 / var.sqrt()).reshape(N, C, H, W)


class EvoNorm2dB0(nn.Module):
    def __init__(self, in_channels, nonlinear=True, momentum=0.9, eps=1e-5):
        super().__init__()
        self.nonlinear = nonlinear
        self.momentum = momentum
        self.eps = eps
        self.gamma = Parameter(torch.Tensor(1, in_channels, 1, 1))
        self.beta = Parameter(torch.Tensor(1, in_channels, 1, 1))
        if nonlinear:
            self.v = Parameter(torch.Tensor(1, in_channels, 1, 1))
        self.register_buffer('running_var', torch.ones(1, in_channels, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        if self.nonlinear:
            init.ones_(self.v)

    def forward(self, x):
        N, C, H, W = x.size()
        if self.training:
            x1 = x.permute(1, 0, 2, 3).reshape(C, -1)
            var = x1.var(dim=1).reshape(1, C, 1, 1)
            self.running_var.copy_(
                self.momentum * self.running_var + (1 - self.momentum) * var)
        else:
            var = self.running_var
        if self.nonlinear:
            den = torch.max((var+self.eps).sqrt(),
                            self.v * x + instance_std(x))
            return x / den * self.gamma + self.beta
        else:
            return x * self.gamma + self.beta


class EvoNorm2dS0(nn.Module):
    def __init__(self, in_channels, groups=8, nonlinear=True):
        super().__init__()
        self.nonlinear = nonlinear
        self.groups = groups
        self.gamma = Parameter(torch.Tensor(1, in_channels, 1, 1))
        self.beta = Parameter(torch.Tensor(1, in_channels, 1, 1))
        if nonlinear:
            self.v = Parameter(torch.Tensor(1, in_channels, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        if self.nonlinear:
            init.ones_(self.v)

    def forward(self, x):
        if self.nonlinear:
            num = torch.sigmoid(self.v * x)
            std = group_std(x, self.groups)
            return num * std * self.gamma + self.beta
        else:
            return x * self.gamma + self.beta


class EvoNorm2d(nn.Module):
    def __init__(self, in_channels, groups=16, norm_type="B0", non_linear=True, momentum=0.9, eps=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.groups = groups
        if norm_type not in ["B0", "S0"]:
            raise ValueError("Invalid EvoNorm version. Choose 'B0' or 'S0' only.")
        self.norm_type = norm_type
        self.non_linear = non_linear
        self.momentum = momentum
        self.eps = eps
        self.evonormb0 = EvoNorm2dB0(self.in_channels, self.non_linear, self.momentum, self.eps)
        self.evonorms0 = EvoNorm2dS0(self.in_channels, self.groups, self.non_linear)

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(x.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.norm_type == "B0":
            return self.evonormb0(x)
        else:
            return self.evonorms0(x)


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x