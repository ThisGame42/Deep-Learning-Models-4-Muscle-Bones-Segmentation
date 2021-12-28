import torch
import torch.nn as nn


class DoubleConvBlock(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv_block(x)


class InputBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InputBlock, self).__init__()
        self.input_block = DoubleConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.input_block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.down_block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_ch, out_ch)
        )

    def forward(self, x):
        return self.down_block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(UpBlock, self).__init__()

        if not bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConvBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutputBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutputBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self,
                 in_channels,
                 classes,
                 base_filter=64):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = classes

        self.inc = InputBlock(in_channels, base_filter)
        self.down1 = DownBlock(base_filter, base_filter * 2)
        self.down2 = DownBlock(base_filter * 2, base_filter * 4)
        self.down3 = DownBlock(base_filter * 4, base_filter * 8)
        self.down4 = DownBlock(base_filter * 8, base_filter * 8)
        self.up1 = UpBlock(base_filter * 16, base_filter * 4)
        self.up2 = UpBlock(base_filter * 8, base_filter * 2)
        self.up3 = UpBlock(base_filter * 4, base_filter)
        self.up4 = UpBlock(base_filter * 2, base_filter)
        self.out_channel = OutputBlock(base_filter, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        up1 = self.up1(x5, x4)
        up2 = self.up2(up1, x3)
        up3 = self.up3(up2, x2)
        features = self.up4(up3, x1)
        logits = self.out_channel(features)
        return {
            "features": features,
            "seg_results": logits
        }