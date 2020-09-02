import torch


# Metrics
def IoU_metric(x, y, smooth = 1.):
    y = y.reshape(x.size())
    intersection = (x * y).sum(dim=1).sum(dim=1).sum(dim=1)
    x_sum = x.sum(dim=1).sum(dim=1).sum(dim=1)
    y_sum = y.sum(dim=1).sum(dim=1).sum(dim=1)
    metric = ((intersection + smooth) / (x_sum + y_sum - intersection + smooth))
    return metric.mean()