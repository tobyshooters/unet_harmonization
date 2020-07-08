# Fully-convolutional U-net for arbitrary input sizes
# Forked from https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256) # 1024 = 512 from down4 + 512 from down3
        self.up2 = Up(512, 128)  # 512 = 256 from up1 + 256 from down2
        self.up3 = Up(256, 64)   # 256 = 128 from up2 + 128 from down1
        self.up4 = Up(128, 64)   # 128 = 64 from up3 + 64 from inc

        self.outc = OutConv(64, n_classes)
        self.tanh = nn.Tanh()


    def forward(self, comp, mask, hist):
        # Compose into (batch x 7 x 512 x 512) input
        x_in = torch.cat([comp, mask, hist], dim=1)

        # Encoder
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.tanh(x)

        # Only apply changes to masked region
        return (1 - mask) * comp + mask * x


# Learn just the residual
# Do everything in LAB-space
# Inject histogram

def inject(layer, histogram):
    h = F.interpolate(histogram, layer.size()[2:], recompute_scale_factor=True)
    return torch.cat([layer, h], dim=1)

class HistNet(nn.Module):
    def __init__(self):
        super(HistNet, self).__init__()

        self.inc = DoubleConv(4, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # +3 for injected histogram
        self.up1 = Up(1024 + 3, 256)
        self.up2 = Up(512 + 3, 128)
        self.up3 = Up(256 + 3, 64)
        self.up4 = Up(128 + 3, 64)

        self.outc = nn.Conv2d(64, 3, kernel_size=1)
        self.tanh = nn.Tanh()


    def forward(self, comp, mask, hist):
        # Compose into (batch x 4 x 512 x 512) input
        x_in = torch.cat([comp, mask], dim=1)

        # Encoder
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, inject(x4, hist))
        x = self.up2(x, inject(x3, hist))
        x = self.up3(x, inject(x2, hist))
        x = self.up4(x, inject(x1, hist))
        x = self.outc(x)

        # Learn the residual, preserves fine-details!
        x = x + comp
        x = self.tanh(x)

        # Only apply changes to masked region
        return (1 - mask) * comp + mask * x
