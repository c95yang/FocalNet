import torch.nn as nn
from .vmamba_layers import *

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False, device='cuda'):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, device=device))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, device=device))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel, device=device))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
            #VSSG(in_chans=in_channel)
        )

    def forward(self, x):
        return self.main(x) + x
        #b, c, h, w = x.shape # torch.Size([4, 32, 256, 256]): (B, C, H, W)
        #self.img_size = (h, w)
        #self.patch_embed = PatchEmbed() #([4, 32, 256, 256]) -> ([4, 256*256, 32]) : (B, C, H, W) -> (B, HW, C)
        #self.patch_unembed = PatchUnEmbed() #([4, 256*256, 32]) -> ([4, 32, 256, 256]) : (B, HW, C) -> (B, C, H, W)

        #res = self.patch_embed(x) #torch.Size([4, 256*256, 32])
        #res = res.reshape(b, self.img_size[0], self.img_size[1], c) #torch.Size([4, 256, 256, 32])
        #res = self.main(res) #torch.Size([4, 256, 256, 32])
        #res.reshape(b, -1, c) #torch.Size([4, 256*256, 32])
        #res = self.patch_unembed(res, self.img_size) #torch.Size([4, 32, 256, 256]): (B, C, H, W)

        #x = res + x #torch.Size([4, 32, 256, 256]): (B, C, H, W)
        #return x

    
