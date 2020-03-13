import numpy as np 


def sum_tensor(input, axes, keepdim=False):
    # 沿着最后一个轴求和
    axes = np.unique(axes)
    if keepdim:
        for ax in axes:
            input = input.sum(ax, keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.sum(ax)
    return input

#对单个样本
def soft_sample_dice(net_output, gt, eps=1e-6):
    axes = tuple(range(2, len(net_output.size())))
    #集合的交集
    intersect = sum_tensor(net_output * gt, axes, keepdim=False)
    # 集合的并集
    denom = sum_tensor(net_output + gt, axes, keepdim=False)
    #dice_loss = 2*|A∩B|/|A∪B|
    return 1 - (2 * intersect.float() / (denom.float() + eps)).mean()

#对一个batch
def soft_batch_dice(net_output, gt, eps=1e-6):
    axes = tuple([0] + list(range(2, len(net_output.size()))))
    intersect = sum_tensor(net_output * gt, axes, keepdim=False)
    denom = sum_tensor(net_output + gt, axes, keepdim=False)
    return 1 - (2 * intersect.float() / (denom.float() + eps)).mean()