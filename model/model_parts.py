import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class SqueezeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, group):
        super().__init__() 
        self.squeeze = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(group, out_channel),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=(2, 1, 1))
            )

    def forward(self, x):
        return self.squeeze(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.resnet_layers = list(self.backbone.children())
        self.layer0 = nn.Sequential(*self.resnet_layers[0:5])
        self.layer1 = nn.Sequential(*self.resnet_layers[5])
        self.layer2 = nn.Sequential(*self.resnet_layers[6])
        self.layer3 = nn.Sequential(*self.resnet_layers[7])

        self.top_layer = nn.Conv2d(2048, 64, kernel_size=1, stride=1, padding=0)
        self.lat_layer2 = nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0)
        self.lat_layer1 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.lat_layer0 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)

        self.smooth4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth16 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        B_size, T_size, Ch_size, H_size, W_size = x.size()
        c = x.reshape(-1, Ch_size, H_size, W_size) # reshape [b, t, c, h, w] --> [b*t, c, h, w]
        c4 = self.layer0(c)
        c8 = self.layer1(c4)
        c16 = self.layer2(c8)
        c32 = self.layer3(c16)

        f32 = self.top_layer(c32)
        f16 = self._upsample_add(f32, self.lat_layer2(c16))
        f8 = self._upsample_add(f16, self.lat_layer1(c8))
        f4 = self._upsample_add(f8, self.lat_layer0(c4))

        f4 = self.smooth4(f4)
        f8 = self.smooth8(f8)
        f16 = self.smooth16(f16)
        f4 = f4.reshape(B_size, T_size, f4.size(1), f4.size(2), f4.size(3)).permute(0, 2, 1, 3, 4)
        f8 = f8.reshape(B_size, T_size, f8.size(1), f8.size(2), f8.size(3)).permute(0, 2, 1, 3, 4)
        f16 = f16.reshape(B_size, T_size, f16.size(1), f16.size(2), f16.size(3)).permute(0, 2, 1, 3, 4)
        f32 = f32.reshape(B_size, T_size, f32.size(1), f32.size(2), f32.size(3)).permute(0, 2, 1, 3, 4)
        return f4, f8, f16, f32


class Decoder(nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        self.squeeze32_0 = SqueezeBlock(in_channel=64, out_channel=64,group= 4)
        self.squeeze32_1 = SqueezeBlock(in_channel=64, out_channel=64, group=2)
        self.squeeze32_2 = SqueezeBlock(in_channel=64, out_channel=64, group=1)

        self.squeeze16_0 = SqueezeBlock(in_channel=64, out_channel=64, group=4)
        self.squeeze16_1 = SqueezeBlock(in_channel=64, out_channel=64, group=2)
        self.conv16_2 = nn.Conv3d(64+64, 64, kernel_size=(1, 1, 1))

        self.squeeze8_0 = SqueezeBlock(in_channel=64, out_channel=64, group=4)
        self.conv8_1 = nn.Conv3d(64+64, 64, kernel_size=(1, 1, 1))

        self.conv4_0 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(8, 64),
            nn.ReLU()
            )
        self.conv4_1 = nn.Conv3d(64+64, 64, kernel_size=(1, 1, 1))
        self.conv4_final = nn.Conv3d(64, out_channel, kernel_size=(1, 1, 1))

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, feats):
        f4, f8, f16, f32 = feats

        f32 = self.squeeze32_0(f32)
        f32 = self.squeeze32_1(f32)
        f32 = self.squeeze32_2(f32)
        f32 = self.upsample(f32)

        f16 = self.squeeze16_0(f16)
        f16 = self.squeeze16_1(f16)
        f16 = torch.cat([f32, f16], dim=1)
        f16 = self.conv16_2(f16)
        f16 = self.upsample(f16)

        f8 = self.squeeze8_0(f8)
        f8 = torch.cat([f16, f8], dim=1)
        f8 = self.conv8_1(f8)
        f8 = self.upsample(f8)

        f4 = self.conv4_0(f4)
        f4 = torch.cat([f8, f4], dim=1)
        f4 = self.conv4_1(f4)
        out = self.conv4_final(f4)
        return out


if __name__ == '__main__':
    images = torch.randn([1, 8, 3, 256, 256]) # t_min = 8
    enc = Encoder()
    f4, f8, f16, f32 = enc(images)
    print("f4.shape: ", f4.shape)
    print("f8.shape: ", f8.shape)
    print("f16.shape: ", f16.shape)
    print("f32.shape: ", f32.shape)
    print("Encoder Done!")
    enc_out = f4, f8, f16, f32
    dec_h = DecoderHeatMap()
    dec_h_out = dec_h(enc_out)
    print("dec_h_out.shape: ", dec_h_out.shape)
    print("DecoderHeatMap Done!")
    dec_e = DecoderEmbedding()
    dec_e_out = dec_e(enc_out)
    print("dec_e_out.shape: ", dec_e_out.shape)
    print("DecoderEmbedding Done!")

