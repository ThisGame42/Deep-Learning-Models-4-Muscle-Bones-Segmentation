import torch
import torch.nn as nn


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channel, out_channels=out_channel,
                               padding=1, kernel_size=3)
        self.conv2 = nn.Conv3d(in_channels=out_channel, out_channels=out_channel,
                               padding=1, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.InstanceNorm3d(num_features=out_channel)
        self.bn2 = nn.InstanceNorm3d(num_features=out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class InputConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(InputConvBlock, self).__init__()
        self.conv = DoubleConvBlock(in_channel, out_channel)

    def forward(self, x):
        return self.conv(x)


class DownConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownConvBlock, self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size=2)
        self.conv = DoubleConvBlock(in_channel, out_channel)

    def forward(self, x):
        x = self.max_pool(x)
        return self.conv(x)


class PadUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, padding):
        super(PadUpBlock, self).__init__()

        self.up = nn.ConvTranspose3d(in_channels=in_ch,
                                     out_channels=in_ch // 2, kernel_size=2, stride=2)

        self.padding = nn.ConstantPad3d(padding=padding, value=0)
        self.double_conv = DoubleConvBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1_padded = self.padding(x1)
        x1_cat = torch.cat([x2, x1_padded], dim=1)
        return self.double_conv(x1_cat)


class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpConvBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels=in_channel,
                                     out_channels=in_channel // 2,
                                     kernel_size=2,
                                     stride=2)

        self.conv = DoubleConvBlock(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class OutConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(OutConvBlock, self).__init__()
        self.out_conv = nn.Conv3d(in_channels=in_channel,
                                  out_channels=out_channel,
                                  kernel_size=1)

    def forward(self, x):
        return self.out_conv(x)


class UNet3D(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 base_filter=32):
        super(UNet3D, self).__init__()

        self.inputConv = InputConvBlock(in_channels, base_filter)

        # Encoder
        self.down1 = DownConvBlock(base_filter, base_filter * 2)
        self.down2 = DownConvBlock(base_filter * 2, base_filter * 4)
        self.down3 = DownConvBlock(base_filter * 4, base_filter * 8)
        self.down4 = DownConvBlock(base_filter * 8, base_filter * 16)

        # Decoder
        self.up1 = UpConvBlock(base_filter * 16, base_filter * 8)
        self.up2 = UpConvBlock(base_filter * 8, base_filter * 4)
        self.up3 = UpConvBlock(base_filter * 4, base_filter * 2)
        self.up4 = UpConvBlock(base_filter * 2, base_filter)

        self.outputConv = OutConvBlock(base_filter, num_classes)

    def forward(self, x):
        x1 = self.inputConv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        up1 = self.up1(x5, x4)
        up2 = self.up2(up1, x3)
        up3 = self.up3(up2, x2)
        up4 = self.up4(up3, x1)
        logits_3d = self.outputConv(up4)
        return {
            "seg_results": logits_3d
        }
