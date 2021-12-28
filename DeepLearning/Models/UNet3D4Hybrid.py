import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channel, out_channels=out_channel,
                               padding=1, kernel_size=3)
        self.conv2 = nn.Conv3d(in_channels=out_channel, out_channels=out_channel,
                               padding=1, kernel_size=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm3d(num_features=out_channel)
        self.bn2 = nn.BatchNorm3d(num_features=out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class InputConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(InputConv, self).__init__()
        self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, x):
        return self.conv(x)


class DownConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownConv, self).__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x


class DownDilatedConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownDilatedConv, self).__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        self.conv = DilatedConvs(in_channel, out_channel)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels=in_channel,
                                     out_channels=in_channel // 2,
                                     kernel_size=2,
                                     stride=2)

        self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(OutConv, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channel,
                               out_channels=in_channel,
                               kernel_size=1)
        self.bn = nn.BatchNorm3d(in_channel)
        self.ac = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=1)

    def forward(self, x):
        return self.conv2(self.ac(self.bn(self.conv1(x))))


class DilatedConvs(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DilatedConvs, self).__init__()
        dilation_rates = [1, 2, 4, 8]

        self.dilatedConv1 = nn.Conv3d(in_channels=in_channel,
                                      out_channels=out_channel,
                                      padding=1,
                                      kernel_size=3,
                                      dilation=dilation_rates[0])
        self.bnd1 = nn.BatchNorm3d(num_features=out_channel)

        self.dilatedConv2 = nn.Conv3d(in_channels=in_channel,
                                      out_channels=out_channel,
                                      padding=2,
                                      kernel_size=3,
                                      dilation=dilation_rates[1])
        self.bnd2 = nn.BatchNorm3d(num_features=out_channel)

        self.dilatedConv3 = nn.Conv3d(in_channels=in_channel,
                                      out_channels=out_channel,
                                      padding=4,
                                      kernel_size=3,
                                      dilation=dilation_rates[2])
        self.bnd3 = nn.BatchNorm3d(num_features=out_channel)

        self.dilatedConv4 = nn.Conv3d(in_channels=in_channel,
                                      out_channels=out_channel,
                                      padding=8,
                                      kernel_size=3,
                                      dilation=dilation_rates[3])
        self.bnd4 = nn.BatchNorm3d(num_features=out_channel)

        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm3d(num_features=out_channel)

        self.conv2 = nn.Conv3d(in_channels=out_channel * 4,
                               out_channels=out_channel,
                               padding=1,
                               kernel_size=3)
        self.bn2 = nn.BatchNorm3d(num_features=out_channel)

    def forward(self, x):
        d1 = F.relu(self.bnd1(self.dilatedConv1(x)), inplace=True)
        d2 = F.relu(self.bnd2(self.dilatedConv2(x)), inplace=True)
        d3 = F.relu(self.bnd3(self.dilatedConv3(x)), inplace=True)
        d4 = F.relu(self.bnd4(self.dilatedConv4(x)), inplace=True)
        features = torch.cat([d1, d2, d3, d4], dim=1)
        out = self.conv2(features)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        return out


class UNetHybrid3D(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 base_filter=32,
                 use_dilation=True):
        super(UNetHybrid3D, self).__init__()

        self.inputConv = InputConv(in_channels, base_filter)

        # Encoder
        self.down1 = DownConv(base_filter, base_filter * 2)
        self.down2 = DownConv(base_filter * 2, base_filter * 4)
        self.down3 = DownConv(base_filter * 4, base_filter * 8)
        # This is the middle layer
        if use_dilation:
            self.down4 = DownDilatedConv(base_filter * 8, base_filter * 16)
        else:
            self.down4 = DownConv(base_filter * 8, base_filter * 16)

        # Decoder
        self.up1 = UpConv(base_filter * 16, base_filter * 8)
        self.up2 = UpConv(base_filter * 8, base_filter * 4)
        self.up3 = UpConv(base_filter * 4, base_filter * 2)
        self.up4 = UpConv(base_filter * 2, base_filter)

        self.outputConv = OutConv(base_filter, num_classes)

    def forward(self, x, feature_2d):
        x1 = self.inputConv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        feature_3d = self.up4(x, x1)
        hybrid_feature = torch.add(feature_2d, feature_3d)
        logits_3d = self.outputConv(hybrid_feature)
        return logits_3d
