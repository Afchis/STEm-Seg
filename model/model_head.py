import math

import torch
import torch.nn as nn

from .model_parts import Encoder, Decoder


class STEmSeg(nn.Module):
    def __init__(self, batch_size, mode, size):
        super().__init__()
        self.size = int(size/4)
        self.b = int(math.ceil(batch_size))
        self.m = mode
        self.xm = torch.linspace(0, 1, self.size).reshape(1, 1, 1, 1, -1).expand(self.b, 1, 8, self.size, self.size)
        self.ym = torch.linspace(0, 1, self.size).reshape(1, 1, 1, -1, 1).expand(self.b, 1, 8, self.size, self.size)
        self.tm = torch.linspace(0, 1, 8).reshape(1, 1, -1, 1, 1).expand(self.b, 1, 8, self.size, self.size)
        self.fm = torch.zeros_like(self.xm)
        self.num = {
            "xyt" : 6,
            "xyf" : 6,
            "xytf" : 8,
            "xyff" : 8
        }
        self.mode = {
            "xyt" : [self.xm, self.ym, self.tm],
            "xyf" : [self.xm, self.ym, self.fm],
            "xytf" : [self.xm, self.ym, self.tm, self.fm],
            "xyff" : [self.xm, self.ym, self.fm, self.fm]
        }
        self.xytm = torch.cat(self.mode[self.m], dim=1)
        self.encoder = Encoder()
        # self.decoder_heatmap = Decoder(out_channel=1)
        self.decoder_embedding = Decoder(out_channel=self.num[self.m]+1)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()



    def forward(self, images):
        num = int(self.num[self.m]/2)
        out = self.encoder(images)
        # Heat_map = self.decoder_heatmap(out)
        Var_Emb = self.decoder_embedding(out)
        Heat_map = Var_Emb[:, :1]
        Var = Var_Emb[:, 1:num+1]
        Emb = Var_Emb[:, num+1:]
        Heat_map = self.sigmoid(Heat_map)
        Var = Var.exp()
        Emb = Emb + self.xytm.cuda().detach()
        return Heat_map, Var, Emb


if __name__ == '__main__':
    images = torch.randn([1, 8, 3, 256, 256])
    model = STEmSeg()
    model(images)

