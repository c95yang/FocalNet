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
    def __init__(self, in_channel, out_channel, dims):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            #BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            #BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
            VSSG(in_chans=in_channel, dims=dims)
        )

    def forward(self, x):
        return self.main(x) + x


    
