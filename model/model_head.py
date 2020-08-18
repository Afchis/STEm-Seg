import torch
import torch.nn as nn

from .model_parts import Encoder, DecoderHeatMap, DecoderEmbedding


class STEmSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder_heatmap = DecoderHeatMap()
        self.decoder_embedding = DecoderEmbedding()

    def forward(self, images):
        out = self.encoder(images)
        heat_map = self.decoder_heatmap(out)
        # print("model_head.heat_map.shape: ", heat_map.shape)
        var = self.decoder_embedding(out)
        # print("model_head.Emb.shape: ", var.shape)
        return heat_map, var


if __name__ == '__main__':
    images = torch.randn([1, 8, 3, 256, 256])
    model = STEmSeg()
    model(images)

