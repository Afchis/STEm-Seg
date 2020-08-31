import torch
import torch.nn as nn

from .model_parts import Encoder, Decoder


class STEmSeg(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.encoder = Encoder()
        self.decoder_heatmap = Decoder(out_channel=1)
        self.decoder_embedding = Decoder(out_channel=8)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        self.b = batch_size
        self.xm = torch.linspace(0, 1, 128).reshape(1, 1, 1, 1, -1).expand(self.b, 1, 8, 128, 128)
        self.ym = torch.linspace(0, 1, 128).reshape(1, 1, 1, -1, 1).expand(self.b, 1, 8, 128, 128)
        self.tm = torch.linspace(0, 1, 8).reshape(1, 1, -1, 1, 1).expand(self.b, 1, 8, 128, 128)
        self.fm = torch.zeros_like(self.xm)
        self.ftyxm = torch.cat([self.xm, self.ym, self.fm, self.fm], dim=1).cuda()

    def forward(self, images):
        out = self.encoder(images)
        Heat_map = self.decoder_heatmap(out)
        Var_Emb = self.decoder_embedding(out)
        # Heat_map = Var_Emb[:, :1]
        Var = Var_Emb[:, :4]
        Emb = Var_Emb[:, 4:]
        Heat_map = self.sigmoid(Heat_map)
        Var = Var.exp()
        Emb = Emb + self.ftyxm.detach()
        return Heat_map, Var, Emb


if __name__ == '__main__':
    images = torch.randn([1, 8, 3, 256, 256])
    model = STEmSeg()
    model(images)

