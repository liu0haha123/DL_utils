import torch
#计算张量在最后一个维度之间的夹角
def angle_last_dim(a, b):
    '''
    Calculate the angle between two nd-arrays (array of vectors) along the last dimension.
    Returns dot product without applying arccos -> higher value = lower angle

    dot product <-> degree conversion: 1->0°, 0.9->23°, 0.7->45°, 0->90°
    By using np.arccos you could return degree in pi (90°: 0.5*pi)

    return: one dimension less than input
    '''
    from .pytorch_einsum import einsum

    if len(a.shape) == 4:
        return torch.abs(einsum('abcd,abcd->abc', a, b) / (torch.norm(a, 2., -1) * torch.norm(b, 2, -1) + 1e-7))
    else:
        return torch.abs(einsum('abcde,abcde->abcd', a, b) / (torch.norm(a, 2., -1) * torch.norm(b, 2, -1) + 1e-7))


def angle_loss(y_pred, y_true):
    """
    Loss based on consine similarity.

    Does not need weighting. y_true is 0 all over background, therefore angle will also be 0 in those areas -> no
    extra masking of background needed.

    Args:
        y_pred: [bs, classes, x, y, z]
        y_true: [bs, classes, x, y, z]

    Returns:
        (loss, None)
    """
    if len(y_pred.shape) == 4:  # 2D
        y_true = y_true.permute(0, 2, 3, 1)
        y_pred = y_pred.permute(0, 2, 3, 1)
    else:  # 3D
        y_true = y_true.permute(0, 2, 3, 4, 1)
        y_pred = y_pred.permute(0, 2, 3, 4, 1)

    nr_of_classes = y_true.shapes[-1]
    scores = torch.zeros(nr_of_classes)

    for idx in range(nr_of_classes):
        angles = angle_last_dim(y_pred, y_true)  # range [0,1], 1 is best

        angles_weighted = angles
        scores[idx] = torch.mean(angles_weighted)

    # doing 1-angle would also work, but 1 will be removed when taking derivatives anyways -> kann simply do *-1
    return -torch.mean(scores), None  # range [0,-1], -1 is best
