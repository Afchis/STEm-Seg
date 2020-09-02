import torch
from .visual_helper import Visual_clusters


class Cluster():
    def __init__(self, vis):
        self.vis = vis

    def _eq2_(self, Emb, Sigma, Myu):
        Sigma = Sigma.view(Emb.size(0), 1, 1, 1)
        Myu = Myu.view(Emb.size(0), 1, 1, 1)
        return ((-0.5) * ((Emb - Myu)**2 * (1/Sigma)).sum(dim=0)).exp()

    def _Sigma_Nyu_(self, Heat_map, Var, Emb):
        if Heat_map.max() < 0.7:
            return False, False
        pos = Heat_map.view(Heat_map.size(0), -1).argmax(dim=1)
        Sigma = Var.view(Var.size(0), Var.size(1), -1)[(torch.arange(Heat_map.size(0))), :, pos]
        Myu = Emb.view(Emb.size(0), Emb.size(1), -1)[(torch.arange(Heat_map.size(0))), :, pos]
        return Sigma, Myu

    def train(self, outs, masks):
        Heat_map, Var, Emb = outs
        pred_list = list()
        pred = torch.zeros([Heat_map.size(2), Heat_map.size(3), Heat_map.size(4)]).cuda()
        for batch in range(Heat_map.size(0)):
            for instance in range(masks[batch].max().item()):
                instance += 1
                masks_j = masks[batch].eq(instance).float().permute(3, 0, 1, 2).detach()
                Var_j, Emb_j = Var[batch]*masks_j, Emb[batch]*masks_j
                Sigma = Var_j.sum(3).sum(2).sum(1) / masks_j.sum(3).sum(2).sum(1)
                Myu = Emb_j.sum(3).sum(2).sum(1) / masks_j.sum(3).sum(2).sum(1)
                pred += self._eq2_(Emb[batch], Sigma, Myu)
            pred_list.append(pred)
        pred_list = torch.stack(pred_list, dim=0)
        return pred_list

    def test(self, outs, iter_in_epoch, iters=7, treshhold=0.5):
        Heat_map, Var, Emb = outs
        unused_masks = torch.ones_like(Heat_map).cuda()
        output_masks = torch.zeros_like(Heat_map).cuda()
        visual_list = list()
        for i in range(iters):
            Sigma, Myu = self._Sigma_Nyu_(Heat_map*unused_masks, Var*unused_masks, Emb*unused_masks)
            if Sigma is False:
                break
            pred = self._eq2_(Emb*unused_masks, Sigma, Myu).view(Heat_map.size())
            visual_list.append(pred.ge(treshhold).float())
            output_masks += pred.ge(treshhold).float()
            unused_masks = unused_masks*pred.lt(treshhold).float()
        if self.vis == True:
            Visual_clusters(visual_list, iter_in_epoch)   
        return output_masks[:, 0]
