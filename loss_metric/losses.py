import torch
import torch.nn.functional as F

from .lovasz_loss import lovasz_hinge


# SmoothLoss:
def _sqrt_sum_(var, var_mean, masks):
    return (((var - var_mean)*masks)**2).sum(3).sum(2).sum(1).mean()

# CenterLoss:
def _eq2_(Emb, Sigma, Myu):
    Sigma = Sigma.view(Emb.size(0), 1, 1, 1)
    Myu = Myu.view(Emb.size(0), 1, 1, 1)
    return ((-0.5) * ((Emb - Myu)**2 * (1/Sigma)).sum(dim=0)).exp()

# EmbeddingLoss:
def _l2_loss_(x, y, smooth = 1.):
    ones = (torch.ones_like(x)).sum()
    out = ((x - y)**2).sum()
    # out = (out + smooth)/(ones + smooth)
    return out

def _dice_loss_(x, y, smooth = 1.):
    y = y.reshape(x.size())
    intersection = (x * y).sum(dim=2).sum(dim=2)
    x_sum = x.sum(dim=2).sum(dim=2)
    y_sum = y.sum(dim=2).sum(dim=2)
    dice_loss = 1 - ((2*intersection + smooth) / (x_sum + y_sum + smooth))
    return dice_loss.mean()

def _IOU_loss_(x, y, smooth = 1.):
    y = y.reshape(x.size())
    intersection = (x * y).sum(dim=1).sum(dim=1).sum(dim=1)
    x_sum = x.sum(dim=1).sum(dim=1).sum(dim=1)
    y_sum = y.sum(dim=1).sum(dim=1).sum(dim=1)
    loss = 1 - ((intersection + smooth) / (x_sum + y_sum - intersection + smooth))
    return loss.mean()


# Losses:
def SmoothLoss(outs, masks, weight=10.):
    _, Var, _ = outs
    loss = 0
    l_i = 0
    for batch in range(masks.size(0)):
        for instance in range(masks[batch].max().item()):
            instance += 1
            masks_j = masks[batch].eq(instance).float().permute(3, 0, 1, 2).detach()
            Var_j = Var[batch] * masks_j
            Var_mean = Var_j.sum(3).sum(2).sum(1) / masks_j.sum()
            loss += _sqrt_sum_(Var_j, Var_mean.view(Var_j.size(0), 1, 1, 1), masks_j) / masks_j.sum()
            l_i += 1
    return weight * loss / l_i

def CenterLoss(outs, masks, weight=1.):
    Heat_map, Var, Emb = outs
    loss = 0
    l_i = 0
    for batch in range(masks.size(0)):
        C = 0
        for instance in range(masks[batch].max().item()):
            instance += 1
            masks_j = masks[batch].eq(instance).float().permute(3, 0, 1, 2)
            Var_j, Emb_j = Var[batch]*masks_j, Emb[batch]*masks_j
            masks_j = masks_j.reshape(masks.size(1), masks.size(2), masks.size(3))
            Sigma = Var_j.sum(3).sum(2).sum(1) / masks_j.sum()
            Nyu = Emb_j.sum(3).sum(2).sum(1) / masks_j.sum()     
            C += _eq2_(Emb[batch], Sigma, Nyu)*masks_j
        loss += F.mse_loss(Heat_map[batch], C.unsqueeze(0).detach())
        l_i += 1
    return weight * loss / l_i

def EmbeddingLoss(pred_masks, masks, weight=1.):
    masks = masks.reshape(pred_masks.size())
    loss = _dice_loss_(pred_masks, masks.detach())
    return weight * loss

def Losses(pred_masks, outs, masks, masks4emb):
    smooth_loss = SmoothLoss(outs, masks)
    center_loss = CenterLoss(outs, masks)
    embedding_loss = EmbeddingLoss(pred_masks, masks4emb)
    total_loss = smooth_loss + center_loss + embedding_loss
    return total_loss, smooth_loss, center_loss, embedding_loss

    # loss = _l2_loss_(pred, label)
    # loss = F.mse_loss(pred, label)
    # loss = _IOU_loss_(pred, label)
    # loss = F.binary_cross_entropy(pred, label)
    # loss = lovasz_hinge(pred, label, per_image=False, ignore=True)