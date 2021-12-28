import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import torchvision.models as models


class Scale(nn.Module):
    """
        Rescale the input to the next layer. See the original paper: https://arxiv.org/abs/1709.07330 if not clear.
    """
    def __init__(self, num_feature):
        super(Scale, self).__init__()
        self.num_feature = num_feature
        self.gamma = nn.Parameter(torch.ones(num_feature), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(num_feature), requires_grad=True)

    def forward(self, x):
        y = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        for i in range(self.num_feature):
            y[:, i, :, :] = x[:, i, :, :].clone() * self.gamma[i] + self.beta[i]
        return y


class Scale3d(nn.Module):
    def __init__(self, num_feature):
        super(Scale3d, self).__init__()
        self.num_feature = num_feature
        self.gamma = nn.Parameter(torch.ones(num_feature), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(num_feature), requires_grad=True)

    def forward(self, x):
        y = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        for i in range(self.num_feature):
            y[:, i, :, :, :] = x[:, i, :, :, :].clone() * self.gamma[i] + self.beta[i]
        return y


class ConvBlock(nn.Sequential):
    def __init__(self, nb_inp_fea, growth_rate, dropout_rate=0):
        super(ConvBlock, self).__init__()
        eps = 1.1e-5
        self.drop = dropout_rate
        self.add_module('norm1', nn.BatchNorm2d(nb_inp_fea, eps=eps, momentum=0.99))
        self.add_module('scale1', Scale(nb_inp_fea))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv2d1', nn.Conv2d(nb_inp_fea, 4 * growth_rate, (1, 1), bias=False))
        self.add_module('norm2', nn.BatchNorm2d(4 * growth_rate, eps=eps, momentum=0.99))
        self.add_module('scale2', Scale(4 * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2d2', nn.Conv2d(4 * growth_rate, growth_rate, (3, 3), padding=(1, 1), bias=False))

    def forward(self, x):
        out = self.norm1(x)
        out = self.scale1(out)
        out = self.relu1(out)
        out = self.conv2d1(out)

        if self.drop > 0:
            out = F.dropout(out, p=self.drop)

        out = self.norm2(out)
        out = self.scale2(out)
        out = self.relu2(out)
        out = self.conv2d2(out)

        if self.drop > 0:
            out = F.dropout(out, p=self.drop)

        return out


class _Transition(nn.Sequential):
    def __init__(self, num_input, num_output, drop=0):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input))
        self.add_module('scale', Scale(num_input))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv2d', nn.Conv2d(num_input, num_output, (1, 1), bias=False))
        if drop > 0:
            self.add_module('drop', nn.Dropout(p=drop, inplace=True))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, nb_filter, growth_rate, dropout_rate=0, weight_decay=1e-4, grow_nb_filters=True):
        super(DenseBlock, self).__init__()
        for i in range(nb_layers):
            layer = ConvBlock(nb_filter + i * growth_rate, growth_rate, dropout_rate)
            self.add_module('denseLayer%d' % (i + 1), layer)

    def forward(self, x):
        features = [x]
        for name, layer in self.named_children():
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)


class DenseUNet2D(nn.Module):
    def __init__(self,
                 block_config=(6, 12, 36, 24),
                 num_classes=14,
                 pretrained=False):
        super(DenseUNet2D, self).__init__()
        self.len_blocks = len(block_config)
        features = models.densenet161(pretrained=pretrained).features
        self.input_block = nn.Sequential(OrderedDict([
            ('conv0_input', nn.Conv2d(1, 3, kernel_size=3, padding=1)),
            ('bn0_input', nn.BatchNorm2d(3)),
            ('relu0_input', nn.ReLU(inplace=True)),
        ]))
        self.encoder = features
        self.decode = nn.Sequential(OrderedDict([
            ('up0', nn.Upsample(scale_factor=2)),
            ('conv2d0', nn.Conv2d(2208, 768, (3, 3), padding=1)),
            ('bn0', nn.BatchNorm2d(768, momentum=1)),
            ('ac0', nn.ReLU(inplace=True)),

            ('up1', nn.Upsample(scale_factor=2)),
            ('zero_padding1', nn.ZeroPad2d(1)),
            ('conv2d1', nn.Conv2d(768, 384, (3, 3), padding=1)),
            ('bn1', nn.BatchNorm2d(384, momentum=1)),
            ('ac1', nn.ReLU(inplace=True)),

            ('up2', nn.Upsample(scale_factor=2)),
            ('conv2d2', nn.Conv2d(384, 96, (3, 3), padding=1)),
            ('bn2', nn.BatchNorm2d(96, momentum=1)),
            ('ac2', nn.ReLU(inplace=True)),

            ('up3', nn.Upsample(scale_factor=2)),
            ('conv2d3', nn.Conv2d(96, 96, (3, 3), padding=1)),
            ('bn3', nn.BatchNorm2d(96, momentum=1)),
            ('ac3', nn.ReLU(inplace=True)),

            ('up4', nn.Upsample(scale_factor=2)),
            ('conv2d4', nn.Conv2d(96, 64, (3, 3), padding=1)),
            ('dropout4', nn.Dropout(p=0.3)),
            ('bn4', nn.BatchNorm2d(64, momentum=1)),
            ('ac4', nn.ReLU(inplace=True))
        ]))
        self.dim_converter = nn.Conv2d(2112, 2208, (1, 1), padding=0)
        self.output = nn.Conv2d(64, num_classes, (1, 1), padding=0)
        self.reshape = nn.Upsample(size=(24, 24), mode="bilinear", align_corners=False)

    def forward(self, x):
        x = self.input_block(x)
        intermediate_results = list()
        feature = x
        for name, layer in self.encoder.named_children():
            feature = layer(feature)
            if name == "relu0":
                intermediate_results.append(feature.clone())
            elif "denseblock" in name:
                if not name == f"denseblock{self.len_blocks}":
                    intermediate_results.append(feature.clone())

        feature = F.relu(feature, inplace=True)
        intermediate_results.append(feature.clone())
        # Expecting to see 4 cached intermediate outputs
        # One from relu0, and each for each dense block (4 in this case).
        assert len(intermediate_results) == self.len_blocks + 1, f"Got {len(intermediate_results)} cached results."
        for name, layer in self.decode.named_children():
            if name == "conv2d0":
                temp_val = self.reshape(intermediate_results[3])
                skip_con = self.dim_converter(temp_val)
                feature = torch.add(skip_con, feature)
            elif name == "conv2d1":
                feature = torch.add(intermediate_results[2], feature)
            elif name == "conv2d2":
                feature = torch.add(intermediate_results[1], feature)
            elif name == "conv2d3":
                feature = torch.add(intermediate_results[0], feature)
            feature = layer(feature)
        logits = self.output(feature)
        return logits, feature


class ConvBlock3D(nn.Sequential):
    def __init__(self, nb_inp_fea, growth_rate, dropout_rate=0., weight_decay=1e-4):
        super(ConvBlock3D, self).__init__()
        eps = 1.1e-5
        self.drop = dropout_rate
        self.add_module('norm1', nn.BatchNorm3d(nb_inp_fea, eps=eps, momentum=1))
        self.add_module('scale1', Scale3d(nb_inp_fea))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv3d1', nn.Conv3d(nb_inp_fea, 4 * growth_rate, (1, 1, 1), bias=False))
        self.add_module('norm2', nn.BatchNorm3d(4 * growth_rate, eps=eps, momentum=1))
        self.add_module('scale2', Scale3d(4 * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv3d2', nn.Conv3d(4 * growth_rate, growth_rate, (3, 3, 3), padding=(1, 1, 1), bias=False))

    def forward(self, x):
        out = self.norm1(x)
        out = self.scale1(out)
        out = self.relu1(out)
        out = self.conv3d1(out)

        if self.drop > 0:
            out = F.dropout(out, p=self.drop)

        out = self.norm2(out)
        out = self.scale2(out)
        out = self.relu2(out)
        out = self.conv3d2(out)

        if self.drop > 0:
            out = F.dropout(out, p=self.drop)

        return out


class DenseBlock3D(nn.Module):
    def __init__(self, nb_layers, nb_filter, growth_rate, dropout_rate=0., weight_decay=1e-4, grow_nb_filters=True):
        super(DenseBlock3D, self).__init__()
        for i in range(nb_layers):
            layer = ConvBlock3D(nb_filter + i * growth_rate, growth_rate, dropout_rate)
            self.add_module('denseLayer3d%d' % (i + 1), layer)

    def forward(self, x):
        features = [x]
        for name, layer in self.named_children():
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)


class _Transition3d(nn.Sequential):
    def __init__(self, num_input, num_output, drop=0):
        super(_Transition3d, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input))
        self.add_module('scale', Scale3d(num_input))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv3d', nn.Conv3d(num_input, num_output, (1, 1, 1), bias=False))
        if drop > 0:
            self.add_module('drop', nn.Dropout(drop, inplace=True))
        self.add_module('pool', nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))


class DenseUNet3D(nn.Module):
    def __init__(self, num_classes, growth_rate=32, block_config=(3, 4, 12, 8), num_init_features=96,
                 drop_rate=0.):
        super(DenseUNet3D, self).__init__()
        nb_filter = num_init_features
        eps = 1.1e-5
        self.drop_rate = drop_rate
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(num_classes + 1, nb_filter, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(nb_filter, eps=eps)),
            ('scale0', Scale3d(nb_filter)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        for i, num_layer in enumerate(block_config):
            block = DenseBlock3D(num_layer, nb_filter, growth_rate, drop_rate)
            nb_filter += num_layer * growth_rate
            self.features.add_module('denseblock3d%d' % (i + 1), block)
            if i != len(block_config) - 1:
                trans = _Transition3d(nb_filter, nb_filter // 2)
                self.features.add_module('transition3d%d' % (i + 1), trans)
                nb_filter = nb_filter // 2

        self.features.add_module('norm5', nn.BatchNorm3d(nb_filter, eps=eps))
        self.features.add_module('scale5', Scale3d(nb_filter))
        self.features.add_module('relu5', nn.ReLU(inplace=True))

        self.decode = nn.Sequential(OrderedDict([
            ('up0', nn.Upsample(scale_factor=(1, 2, 2))),
            ('conv2d0', nn.Conv3d(nb_filter, 504, (3, 3, 3), padding=1)),
            ('bn0', nn.BatchNorm3d(504, momentum=1)),
            ('ac0', nn.ReLU(inplace=True)),

            ('up1', nn.Upsample(scale_factor=(1, 2, 2))),
            ('zero_padding1', nn.ZeroPad2d(1)),
            ('conv2d1', nn.Conv3d(504, 224, (3, 3, 3), padding=1)),
            ('bn1', nn.BatchNorm3d(224, momentum=1)),
            ('ac1', nn.ReLU(inplace=True)),

            ('up2', nn.Upsample(scale_factor=(1, 2, 2))),
            ('conv2d2', nn.Conv3d(224, 192, (3, 3, 3), padding=1)),
            ('bn2', nn.BatchNorm3d(192, momentum=1)),
            ('ac2', nn.ReLU(inplace=True)),

            ('up3', nn.Upsample(scale_factor=(2, 2, 2))),
            ('conv2d3', nn.Conv3d(192, 96, (3, 3, 3), padding=1)),
            ('bn3', nn.BatchNorm3d(96, momentum=1)),
            ('ac3', nn.ReLU(inplace=True)),

            ('up4', nn.Upsample(scale_factor=(2, 2, 2))),
            ('conv2d4', nn.Conv3d(96, 64, (3, 3, 3), padding=1)),
            ('bn4', nn.BatchNorm3d(64, momentum=1)),
            ('ac4', nn.ReLU(inplace=True))
        ]))

        self.finalConv3d1 = nn.Conv3d(64, 64, (3, 3, 3), padding=(1, 1, 1))
        self.finalBn = nn.BatchNorm3d(64)
        self.finalAc = nn.ReLU(inplace=True)
        self.finalConv3d2 = nn.Conv3d(64, num_classes, kernel_size=1)

    def forward(self, x, feature_2d):
        out = self.features(x)
        feature_3d = self.decode(out)

        fused_feature = torch.add(feature_3d, feature_2d)
        feature = self.finalConv3d1(fused_feature)
        if self.drop_rate > 0.:
            feature = F.dropout(feature, p=self.drop_rate)

        feature = self.finalBn(feature)
        feature = self.finalAc(feature)
        final_out = self.finalConv3d2(feature)
        return final_out