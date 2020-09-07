import torch


def _IoU_metric_(x, y, smooth = 1.):
    y = y.reshape(x.size())
    intersection = (x * y).sum(dim=1).sum(dim=1).sum(dim=1)
    x_sum = x.sum(dim=1).sum(dim=1).sum(dim=1)
    y_sum = y.sum(dim=1).sum(dim=1).sum(dim=1)
    metric = ((intersection + smooth) / (x_sum + y_sum - intersection + smooth))
    return metric.mean()
# Metrics
def IoU_metric(pred_masks, masks, smooth = 1.):
    with torch.no_grad():
        metric = 0
        for batch in range(masks.size(0)):
            masks_j = list()
            for instance in range(masks[batch].max().item()):
                instance += 1
                masks_j.append(masks[batch].eq(instance).float().permute(3, 0, 1, 2))
            masks_j = torch.cat(masks_j, dim=0)
            metric += _IoU_metric_((pred_masks[batch]>0.5).float(), masks_j)
    return (metric / masks.size(0)).item()