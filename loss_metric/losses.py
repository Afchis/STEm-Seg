import torch
import torch.nn.functional as F

from .lovasz_loss import lovasz_hinge


def _emb_(Emb):
    xm = torch.linspace(0, 1, 128).reshape(1, 1, 1, 1, -1).expand(Emb.size(0), 1, 8, 128, 128)
    ym = torch.linspace(0, 1, 128).reshape(1, 1, 1, -1, 1).expand(Emb.size(0), 1, 8, 128, 128)
    tm = torch.linspace(0, 1, 8).reshape(1, 1, -1, 1, 1).expand(Emb.size(0), 1, 8, 128, 128)
    tyxm = torch.cat([xm, ym, tm], dim=1).cuda()
    return Emb + tyxm.detach()

# SmoothLoss:
def _sqrt_sum_(var, var_mean, masks4):
    return (((var - var_mean)*masks4)**2).sum(4).sum(3).sum(2).mean()

# CenterLoss:
def _eq2_(Emb, Sigma, Nyu):
    A = 1 / (torch.sqrt(torch.tensor(2*3.1415))**Emb.size(1) * torch.sqrt(torch.prod(Sigma, dim=1)))
    Sigma = Sigma.view(Emb.size(0), Emb.size(1), 1, 1, 1)
    Nyu = Nyu.view(Emb.size(0), Emb.size(1), 1, 1, 1)
    B = (-0.5) * ((Emb - Nyu)**2 * (1/Sigma)).sum(dim=1)
    A = A.view(B.size(0), 1, 1, 1)
    return A*torch.exp(B)

# EmbeddingLoss:
def l2_loss(x, y, smooth = 1.):
    ones = (torch.ones_like(x)).sum()
    out = ((x - y)**2).sum()
    # out = (out + smooth)/(ones + smooth)
    return out

def dice_loss(x, y, smooth = 1.):
    y = y.reshape(x.size())
    intersection = (x * y).sum(dim=2).sum(dim=2)
    x_sum = x.sum(dim=2).sum(dim=2)
    y_sum = y.sum(dim=2).sum(dim=2)
    dice_loss = 1 - ((2*intersection + smooth) / (x_sum + y_sum + smooth))
    return dice_loss.mean()

def cross_entropy_loss(x, y):
    cross_entropy = torch.nn.CrossEntropyLoss()
    return cross_entropy(x, y)


# Losses:
def SmoothLoss(outs, masks4, weight=10.):
    _, Var, _ = outs
    Var_j = Var * masks4
    Var_mean = Var_j.sum(4).sum(3).sum(2) / masks4.sum(4).sum(3).sum(2)
    loss = _sqrt_sum_(Var_j, Var_mean.view(Var_j.size(0), Var_j.size(1), 1, 1, 1), masks4) / masks4.sum()
    return weight * loss

def CenterLoss(outs, masks4, weight=1.):
    Heat_map, Var, Emb = outs
    Emb = _emb_(Emb)
    Heat_map_j, Var_j, Emb_j = Heat_map*masks4, Var*masks4, Emb*masks4
    Sigma = Var_j.sum(4).sum(3).sum(2) / masks4.sum(4).sum(3).sum(2)
    Nyu = Emb_j.sum(4).sum(3).sum(2) / masks4.sum(4).sum(3).sum(2)
    C_j = _eq2_(Emb, Sigma, Nyu)
    C_j = C_j.detach()
    # loss = ((C_j - Heat_map.squeeze())**2).sum(3).sum(2).sum(1) / torch.ones_like(masks4).sum(4).sum(3).sum(2).squeeze()
    loss = F.mse_loss(Heat_map.squeeze(), C_j)
    return weight * loss

def EmbeddingLoss(pred, label, weight=1.):
    label = label.reshape(pred.size())
    # loss = F.mse_loss(pred, label)
    # loss = dice_loss(pred, label)
    # loss = F.binary_cross_entropy(pred, label)
    loss = lovasz_hinge(pred, label, per_image=False, ignore=None)
    return weight * loss
