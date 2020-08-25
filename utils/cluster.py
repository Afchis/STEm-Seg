import torch


class Cluster:
    '''
    TODO:
    *** make utils def _eq2_() or class _eq2_()
    '''
    def _eq2_(self, Emb, Sigma, Nyu):
        A = 1 / (torch.sqrt(torch.tensor(2*3.1415))**Emb.size(1) * torch.sqrt(torch.prod(Sigma, dim=1)))
        Sigma = Sigma.view(Emb.size(0), Emb.size(1), 1, 1)
        Nyu = Nyu.view(Emb.size(0), Emb.size(1), 1, 1)
        B = (-0.5) * ((Emb - Nyu)**2 * (1/Sigma)).sum(dim=1)
        A = A.view(B.size(0), 1, 1, 1)
        return A*torch.exp(B)

    def run(self, outs, masks4):
        Heat_map, Var, Emb = outs
        Var, Emb = Var*masks4, Emb*masks4
        Sigma = Var.sum(3).sum(2) / masks4.sum(3).sum(2)
        Nyu = Emb.sum(3).sum(2) / masks4.sum(3).sum(2)
        pred = self._eq2_(Emb, Sigma, Nyu)
        pred = (pred > 0.5).float()
        return pred

