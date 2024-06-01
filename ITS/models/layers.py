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
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False),
            # BasicConv_G(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            # BasicConv_G(out_channel, out_channel, kernel_size=3, stride=1, relu=False),
            # VSSG(in_chans=in_channel)
        )

    def forward(self, x):
        return self.main(x) + x
    
class BasicConv_G(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv_G, self).__init__()

        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=biasc))
        else:
            layers.append(GhostModule(in_channel, out_channel, stride=stride, relu=False))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel, device='cuda'))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)


    def forward(self, x):
        y = self.main(x) #([32, 3, 128, 128])->([32, 16, 128, 128])
        return y

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, transpose=False):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False, device='cuda'),
            # nn.BatchNorm2d(init_channels),
            # nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False, device='cuda'),
            # nn.BatchNorm2d(new_channels),
            # nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


    
