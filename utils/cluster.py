import torch


class Cluster:
    def _emb_(self, Emb):
        xm = torch.linspace(0, 1, 128).reshape(1, 1, 1, 1, -1).expand(Emb.size(0), 1, 8, 128, 128)
        ym = torch.linspace(0, 1, 128).reshape(1, 1, 1, -1, 1).expand(Emb.size(0), 1, 8, 128, 128)
        tm = torch.linspace(0, 1, 8).reshape(1, 1, -1, 1, 1).expand(Emb.size(0), 1, 8, 128, 128)
        tyxm = torch.cat([xm, ym, tm], dim=1).cuda()
        return Emb + tyxm

    def _eq2_(self, Emb, Sigma, Nyu):
        A = 1 / (torch.sqrt(torch.tensor(2*3.1415))**Emb.size(1) * torch.sqrt(torch.prod(Sigma, dim=1)))
        Sigma = Sigma.view(Emb.size(0), Emb.size(1), 1, 1, 1)
        Nyu = Nyu.view(Emb.size(0), Emb.size(1), 1, 1, 1)
        B = (-0.5) * ((Emb - Nyu)**2 * (1/Sigma)).sum(dim=1)
        A = A.view(B.size(0), 1, 1, 1)
        return A*torch.exp(B)

    def _Sigma_Nyu_(self, Heat_map, Var, Emb):
        if Heat_map.max() < 0.5:
            return False, False
        pos = Heat_map.view(Heat_map.size(0), -1).argmax(dim=1)
        Sigma = Var.view(Var.size(0), Var.size(1), -1)[(torch.arange(Heat_map.size(0))), :, pos]
        Nyu = Emb.view(Emb.size(0), Emb.size(1), -1)[(torch.arange(Heat_map.size(0))), :, pos]
        return Sigma, Nyu

    def run(self, outs, masks4):
        Heat_map, Var, Emb = outs
        Emb = self._emb_(Emb)
        Var_m, Emb_m = Var*masks4, Emb*masks4
        Sigma = Var_m.sum(4).sum(3).sum(2) / masks4.sum(4).sum(3).sum(2)
        Nyu = Emb_m.sum(4).sum(3).sum(2) / masks4.sum(4).sum(3).sum(2)
        pred = self._eq2_(Emb, Sigma, Nyu)
        return pred

    def test_run(self, outs, iters=1):
        Heat_map, Var, Emb = outs
        Emb = self._emb_(Emb)
        unused_masks = torch.ones_like(Heat_map).cuda()
        output_masks = torch.zeros_like(Heat_map).cuda()
        for i in range(iters):
            Sigma, Nyu = self._Sigma_Nyu_(Heat_map*unused_masks, Var*unused_masks, Emb*unused_masks)
            if Sigma is False:
                break
            pred = self._eq2_(Emb, Sigma, Nyu).view(Heat_map.size())
            output_masks = output_masks + pred.ge(0.5).float()
            unused_masks = unused_masks * output_masks.eq(0).float()        
        return output_masks[:, 0]
