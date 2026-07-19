import torch
import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.ops import StochasticDepth

from models.decor import DecorConv2d


class Block(nn.Module):
    def __init__(self, in_channels, stochastic_depth_prob):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=in_channels,
                                        kernel_size=7,
                                        groups=in_channels,
                                        padding=3)

        self.layer_norm = nn.LayerNorm(normalized_shape=in_channels, eps=1e-6)

        self.pointwise_conv1 = DecorConv2d(nn.Conv2d, in_channels=in_channels, out_channels=in_channels * 4, kernel_size=1)

        self.gelu = nn.GELU()

        self.pointwise_conv2 = DecorConv2d(nn.Conv2d, in_channels=in_channels * 4, out_channels=in_channels, kernel_size=1)

        self.layer_scale = nn.Parameter(torch.full((in_channels, 1, 1), 1e-6))

        self.stochastic_depth_prob = stochastic_depth_prob
        self.stochastic_depth = StochasticDepth(p=self.stochastic_depth_prob, mode="row")

    def forward(self, x):
        residual = x
        x = self.depthwise_conv(x)
        x = self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.pointwise_conv1(x)
        x = self.gelu(x)
        x = self.pointwise_conv2(x)
        x = self.layer_scale * x
        x = self.stochastic_depth(x)
        return x + residual


class DownSampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=in_channels, eps=1e-6)
        self.conv = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=2,
                                stride=2)

    def forward(self, x):
        x = self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.conv(x)


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=4,
                                stride=4)
        self.layer_norm = nn.LayerNorm(normalized_shape=out_channels, eps=1e-6)

    def forward(self, x):
        return self.layer_norm(self.conv(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNext(nn.Module):
    def __init__(self, in_channels=3, channels=[96, 192, 384, 768], blocks=[3, 3, 9, 3], 
                stochastic_depth_prob=0.1, num_classes=1000):
        super().__init__()

        self.stem = Stem(in_channels=in_channels, out_channels=channels[0])

        self.stage1 = self._make_stage(in_channels=channels[0], num_blocks=blocks[0], stochastic_depth_prob=stochastic_depth_prob)
        self.downsample1 = DownSampler(in_channels=channels[0], out_channels=channels[1])

        self.stage2 = self._make_stage(in_channels=channels[1], num_blocks=blocks[1], stochastic_depth_prob=stochastic_depth_prob)
        self.downsample2 = DownSampler(in_channels=channels[1], out_channels=channels[2])

        self.stage3 = self._make_stage(in_channels=channels[2], num_blocks=blocks[2], stochastic_depth_prob=stochastic_depth_prob)
        self.downsample3 = DownSampler(in_channels=channels[2], out_channels=channels[3])

        self.stage4 = self._make_stage(in_channels=channels[3], num_blocks=blocks[3], stochastic_depth_prob=stochastic_depth_prob)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head_norm = nn.LayerNorm(normalized_shape=channels[-1], eps=1e-6)
        self.classifier = nn.Linear(in_features=channels[-1], out_features=num_classes)

        self.apply(self._init_weights)

    def _make_stage(self, in_channels, num_blocks, stochastic_depth_prob):
        layers = [Block(in_channels=in_channels, stochastic_depth_prob=stochastic_depth_prob)
                for i in range(num_blocks)]
        return nn.Sequential(*layers)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample1(x)
        x = self.stage2(x)
        x = self.downsample2(x)
        x = self.stage3(x)
        x = self.downsample3(x)
        x = self.stage4(x)
        x = self.avg_pool(x)
        x = self.head_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.classifier(x.reshape(x.shape[0], -1))