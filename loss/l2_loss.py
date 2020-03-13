import torch


def l2_loss(y_pred, y_true, weights=None):
    """
    Calculate the euclidian distance (=l2 norm / frobenius norm) between tensors.
    Expects a tensor image as input (6 channels per class).

    Args:
        y_pred: [bs, classes, x, y, z]
        y_true: [bs, classes, x, y, z]
        weights: None, just for keeping the interface the same for all loss functions

    Returns:
        loss
    """
    if len(y_pred.shape) == 4:  # 2D
        y_true = y_true.permute(0, 2, 3, 1)
        y_pred = y_pred.permute(0, 2, 3, 1)
    else:  # 3D
        y_true = y_true.permute(0, 2, 3, 4, 1)
        y_pred = y_pred.permute(0, 2, 3, 4, 1)

    nr_of_classes = y_true.shape[-1]
    scores = torch.zeros(nr_of_classes)

    for idx in range(nr_of_classes):
        dist = torch.dist(y_pred, y_true, 2)  # calc l2 norm / euclidian distance / frobenius norm
        scores[idx] = torch.mean(dist)

    return torch.mean(scores), None