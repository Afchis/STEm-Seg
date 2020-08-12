import torch
import torch.nn as nn

from model_parts import Encoder, DecoderHeatMap


class STEmSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder_heatmap = DecoderHeatMap()

    def forward(self, images):
        out = self.encoder(images)
        return out


if __name__ == '__main__':
    images = torch.randn([1, 5, 3, 256, 256])
    model = STEmSeg()
    f4, f8, f16, f32 = model(images)
    print("f4.shape: ", f4.shape)
    print("f8.shape: ", f8.shape)
    print("f16.shape: ", f16.shape)
    print("f32.shape: ", f32.shape)