import torch
from torch import nn

from models.decor import DecorConv2d


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            DecorConv2d(nn.Conv2d, gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_conv = nn.Sequential(
            DecorConv2d(nn.Conv2d, skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            DecorConv2d(nn.Conv2d, inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        attention = self.psi(self.relu(self.gate_conv(gate) + self.skip_conv(skip)))
        return skip * attention


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, channels=[32, 64, 128, 256, 512], num_classes=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = conv_block(in_channels, channels[0])
        self.enc2 = conv_block(channels[0], channels[1])
        self.enc3 = conv_block(channels[1], channels[2])
        self.enc4 = conv_block(channels[2], channels[3])
        self.enc5 = conv_block(channels[3], channels[4])

        self.up4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2)
        self.att4 = AttentionGate(channels[3], channels[3], channels[3] // 2)
        self.dec4 = conv_block(channels[3] * 2, channels[3])

        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.att3 = AttentionGate(channels[2], channels[2], channels[2] // 2)
        self.dec3 = conv_block(channels[2] * 2, channels[2])

        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.att2 = AttentionGate(channels[1], channels[1], channels[1] // 2)
        self.dec2 = conv_block(channels[1] * 2, channels[1])

        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.att1 = AttentionGate(channels[0], channels[0], channels[0] // 2)
        self.dec1 = conv_block(channels[0] * 2, channels[0])

        self.final_conv = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        d4 = self.up4(e5)
        s4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([s4, d4], dim=1))

        d3 = self.up3(d4)
        s3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([s3, d3], dim=1))

        d2 = self.up2(d3)
        s2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([s2, d2], dim=1))

        d1 = self.up1(d2)
        s1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([s1, d1], dim=1))

        return self.final_conv(d1)
