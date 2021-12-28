import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropy(nn.Module):
    """
        Cross entropy that accepts soft targets
    """

    def __init__(self):
        super(SoftCrossEntropy, self).__init__()
        self.weights = torch.ones(14)

        for i in range(14):
            if i == 0:
                self.weights[i] = 0.01
            else:
                self.weights[i] = 0.99 / 13

    def forward(self,
                raw_pred: torch.Tensor,
                ref_labels: torch.Tensor):
        log_prob = F.log_softmax(raw_pred, dim=1)
        res = - (ref_labels * log_prob)
        ret_val = torch.mean(torch.sum(res, dim=1))
        return ret_val
