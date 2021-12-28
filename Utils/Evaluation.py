import torch


def compute_DSC(predictions: torch.Tensor,
                ref_labels: torch.Tensor,
                weights: torch.Tensor = None,
                epsilon=1.e-6,
                use_vnet_dice=True) -> torch.Tensor:
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

        This function computes the VNet Dice if use_vnet_dice is True, and computes the standard Dice otherwise.

    :param predictions: Output of a model. Assumed to be in probabilities (i.e. in [0, 1], not in logits!)
    :param ref_labels: One-hot encoded reference labels
    :param weights: Cx1 tensor of weight per class
    :param epsilon: Prevents division by zero error
    :param use_vnet_dice: See above
    :return: A torch Tensor containing average (over the batch) Dice coefficients for each class.
    """
    assert predictions.size() == ref_labels.size(), "The predictions and the reference labels are not in the same shape"

    assert predictions.dim() == 5 or predictions.dim() == 4, f"Only 4D or 5D predictions are supported"
    assert torch.max(predictions) <= 1. and torch.min(predictions) >= 0., "Invalid values in predictions detected"
    assert torch.max(ref_labels) <= 1 and torch.min(ref_labels) >= 0, f"Invalid values in reference labels detected"

    prob_flatten = flatten(predictions)
    ref_flatten = flatten(ref_labels).float()

    # compute per channel Dice Coefficient
    intersect = (prob_flatten * ref_flatten).sum(-1)
    if weights is not None:
        intersect *= weights

    # here we can use standard dice (input + target).sum(-1)
    # or extension (see V-Net) (input^2 + target^2).sum(-1)
    if use_vnet_dice:
        denominator = (prob_flatten * prob_flatten).sum(-1) + (ref_flatten * ref_flatten).sum(-1)
    else:
        denominator = prob_flatten.sum(-1) + ref_flatten.sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


def flatten(tensor: torch.Tensor) -> torch.Tensor:
    """
    Flattens a given tensor into a two-dimensional tensor. Class channel becomes the first dimension and
    other dimensions are squashed into one.

       3D: (Batch, Class, Depth, Height, Width) -> (C, B * D * H * W)\n
       2D: (B, C, H, W) -> (C, B * H * W)
    """

    num_classes = tensor.size()[1]
    # new dimension order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(num_classes, -1)
