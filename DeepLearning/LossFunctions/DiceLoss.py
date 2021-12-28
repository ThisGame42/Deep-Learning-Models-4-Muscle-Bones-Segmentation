import torch
import torch.nn as nn

from Utils.Evaluation import compute_DSC


class WeightedDiceLoss(nn.Module):
    """
        Dice loss function that optimises directly against the DSC score.
    """
    def __init__(self,
                 num_classes: int = 3,
                 ignore_background: bool = False,
                 use_softmax: bool = True):

        super(WeightedDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        print(f"Dice loss, is background ignored -> {self.ignore_background}.")
        if use_softmax:
            self.probs_fn = nn.Softmax(dim=1)
        else:
            self.probs_fn = nn.Sigmoid()

    def forward(self,
                raw_pred: torch.Tensor,
                ref_labels: torch.Tensor):
        # convert the raw output from the network (which is in logits) to probabilities (i.e., in [0, 1])
        probs = self.probs_fn(raw_pred)
        dsc_scores = compute_DSC(predictions=probs,
                                 ref_labels=ref_labels)
        # if self.ignore_background is True, we ignore the dsc score for the background class
        # when calculating the mean dsc scores of this batch
        dsc_scores = dsc_scores[1:] if self.ignore_background else dsc_scores
        loss_val = 1 - torch.mean(dsc_scores, dim=0)
        return loss_val
