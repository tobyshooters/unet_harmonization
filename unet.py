# Fully-convolutional U-net for arbitrary input sizes
# Based on https://github.com/milesial/Pytorch-UNet
# Based on https://github.com/LeeJunHyun/Image_Segmentation

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
    # Since bilinear upsampling preserves number of channels,
    # we must decrease number of channels in the double convolution.
    # Were we to use ConvTranpose, the channels would be reduced
    # while upsampling, but before concatenating.
    # (https://github.com/milesial/Pytorch-UNet/issues/69)k

    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

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

        # Learn the residual
        return comp + mask * x


class Attention(nn.Module):
    """Attention module for a U-net"""

    def __init__(self, Fg, Fl, Fi):
        # gate, layer, intermediate
        super(Attention, self).__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(Fg, Fi, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(Fi)
        )
        self.Wx = nn.Sequential(
            nn.Conv2d(Fl, Fi, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(Fi)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(Fi, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_up, x_skip):
        g = self.Wg(x_up)
        x = self.Wx(x_skip)
        j = self.relu(g + x)
        p = self.psi(j)
        return p * x_skip


class UpAttention(nn.Module):
    """Upsamples input, gates the skip features, and concatenates"""

    def __init__(self, up_channels, skip_channels, int_channels, out_channels):
        super().__init__()
        in_channels = up_channels + skip_channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att = Attention(up_channels, skip_channels, int_channels)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x_up, x_skip):
        # Upsample
        x_up = self.up(x_up)
        x_att = self.att(x_up, x_skip)

        # Pad and concatenate
        diffY = x_att.size()[2] - x_up.size()[2]
        diffX = x_att.size()[3] - x_up.size()[3]
        x_up = F.pad(x_up, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x_att, x_up], dim=1)

        # Convolve
        return self.conv(x)


class AttentionUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = UpAttention(up_channels=512, skip_channels=512, int_channels=256, out_channels=256)
        self.up2 = UpAttention(up_channels=256, skip_channels=256, int_channels=128, out_channels=128)
        self.up3 = UpAttention(up_channels=128, skip_channels=128, int_channels=64,  out_channels=64)
        self.up4 = UpAttention(up_channels=64,  skip_channels=64,  int_channels=32,  out_channels=64)

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
        d5 = self.up1(x5, x4)
        d4 = self.up2(d5, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up4(d3, x1)
        d1 = self.outc(d2)
        x = self.tanh(d1)

        # Learn the residual
        return comp + mask * x


def inject(layer, histogram):
    h = F.interpolate(histogram, layer.size()[2:], recompute_scale_factor=True)
    return torch.cat([layer, h], dim=1)

class HistNet(nn.Module):
    """Inject color at all levels, based on PaletteNet"""

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
