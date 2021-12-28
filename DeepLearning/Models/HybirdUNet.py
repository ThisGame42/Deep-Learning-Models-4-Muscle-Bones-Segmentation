import torch
import torch.nn as nn
import torch.nn.functional as F

from DeepLearning.Models.UNet import UNet
from DeepLearning.Models.UNet3D4Hybrid import UNetHybrid3D


class HybridUNet(nn.Module):
    def __init__(self, model2d: UNet, model3d: UNetHybrid3D):
        """
            Expects at least model2d is pre-trained
        """
        super(HybridUNet, self).__init__()
        self.model2d = model2d
        self.model3d = model3d

    def forward(self, x):
        input_2d = torch.unsqueeze(x[0, 0, ...], dim=1)
        ret2d = self.model2d(input_2d)
        logits2d, features2d = ret2d["seg"], ret2d["features"]
        output2d = F.softmax(logits2d, dim=1)
        output2d = output2d.permute(1, 0, 2, 3)
        feature2d = features2d.permute(1, 0, 2, 3)
        output2d = torch.unsqueeze(output2d, dim=0)
        feature2d = torch.unsqueeze(feature2d, dim=0)
        input3d = torch.cat([output2d, x], dim=1)
        return self.model3d(input3d, feature2d)