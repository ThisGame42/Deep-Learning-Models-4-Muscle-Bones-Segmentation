import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.Evaluation import flatten


class MultiClassCSE(nn.Module):
    def __init__(self, weights=None, num_classes=14):
        super(MultiClassCSE, self).__init__()
        self.num_classes = num_classes
        # if weights is None:
        #     weights = torch.ones(num_classes) / num_classes
        # assert torch.sum(weights) == 1.
        # self.weights = weights

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)

        probs_f = flatten(inputs)
        target_f = flatten(targets)

        cse_loss = torch.zeros(self.num_classes)
        for i in range(probs_f.size()[0]):
            cse_loss[i] = F.binary_cross_entropy(probs_f[i], target_f[i])
        return torch.mean(cse_loss)

