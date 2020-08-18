import torch

class Cluster():
    def __init__(self):
        '''
        TODO:
        ** batch size for self.tyxm
        '''
        self.xm = torch.linspace(0, 1, 64).reshape(1, 1, 1, 1, -1).expand(1, 1, 8, 64, 64)
        self.ym = torch.linspace(0, 1, 64).reshape(1, 1, 1, -1, 1).expand(1, 1, 8, 64, 64)
        self.tm = torch.linspace(0, 1, 8).reshape(1, 1, -1, 1, 1).expand(1, 1, 8, 64, 64)
        # print("torch.cat.shapes: ", self.xm.shape, self.ym.shape, self.tm.shape)
        self.tyxm = torch.cat([self.xm, self.ym, self.tm], dim=1)
        # print("self.tyxm.shape: ", self.tyxm.shape)

    def _SigmaNyu_(self, outs):
        heat_map, var = outs
        # print("var.shape: ", var.shape, "self.tyxm.shape: ", self.tyxm.shape)
        emb = var + self.tyxm
        # print("heat_map.shape: ", heat_map.shape)
        idx = (heat_map == torch.max(heat_map))[0, 0].nonzero().tolist()[0]
        # print("idx_max:", idx)
        SIGMA = var[:, :, idx[0], idx[1], idx[2]]
        Nyu = emb[:, :, idx[0], idx[1], idx[2]]
        # print("emb.shape:", emb.shape, "SIGMA.shape:", SIGMA.shape, "Nyu.shape:", Nyu.shape)
        return emb, SIGMA, Nyu

    def _eq2_(self, emb, SIGMA, Nyu):
        '''
        TODO:
        * change 3.14 on pi
        '''
        emb = emb.permute(0, 2, 3, 4, 1).unsqueeze(5)
        # print("emb.shape.PERMUTE: ", emb.shape)
        for batch in range(SIGMA.size(0)):
            Sigma = torch.diag(SIGMA[batch])            
            # print("Sigma.shape: ", Sigma.shape)
            a = (1/(torch.sqrt(torch.tensor(2*3.1415))**3*torch.sqrt(torch.det(Sigma))))
            # print("emb[batch].shape: ", emb[batch].shape, "Nyu[batch].shape: ", Nyu[batch].unsqueeze(1).shape)
            # print("(emb[batch] - Nyu[batch]).shape: ", (emb[batch] - Nyu[batch].unsqueeze(1)).shape)
            transpose = torch.transpose((emb[batch] - Nyu[batch].unsqueeze(1)), dim0=3, dim1=4)
            # print("################transpose.shape: ", transpose.shape)
            inverse = torch.inverse(Sigma)#.unsqueeze(0)
            # print("inverse.shape", inverse.shape)
            minus = (emb[batch] - Nyu[batch].unsqueeze(1))
            # print("minus.shape", minus.shape)
            # print("(transpose@inverse@minus).shape: ", (transpose@inverse@minus).shape)
            exp = torch.exp(-1/2*transpose@inverse@minus)
            # print("exp.shape: ", exp.shape)
            # print("a.shape: ", a)
        return a*exp

    def cluster(self, outs):
        emb, SIGMA, Nyu = self._SigmaNyu_(outs)
        p = self._eq2_(emb, SIGMA, Nyu)
        return p

