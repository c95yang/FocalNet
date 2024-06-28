import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res):
        super(EBlock, self).__init__()
        #self.scale = nn.Parameter(torch.ones(out_channel,1,1, device='cuda'))

        # depth [2] 
        layers = [VSSG(gl_merge=False, in_chans=out_channel, patch_size_global=2, patch_size_local=2, forward_type="v4", mlp_ratio=1.0) for _ in range(num_res)]
        
        #layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        res = self.layers(x)
        #return self.scale * res + x #scaled residual
        return res + x
    
    def flops(self, x):
        flops = 0
        for layer in self.layers:
            flops += VSSG.flops(self, x)
            x = layer(x)
        return flops


class DBlock(nn.Module):
    def __init__(self, channel, num_res):
        super(DBlock, self).__init__()
        #self.scale = nn.Parameter(torch.ones(channel,1,1, device='cuda'))

        # depth [2] 
        layers = [VSSG(gl_merge=False, in_chans=channel, patch_size_global=2, patch_size_local=2, forward_type="v4", mlp_ratio=1.0) for _ in range(num_res)]

        #layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        res = self.layers(x)
        #return self.scale * res + x #scaled residual
        return res + x 
    
    def flops(self, x):
        flops = 0
        for layer in self.layers:
            flops += VSSG.flops(self, x)
            x = layer(x)
        return flops


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True, device='cuda')
        )
    def forward(self, x):
        x = self.main(x)
        return x

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)
    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))

class MIMOUNet(nn.Module):
    def __init__(self, num_res=1):
        super(MIMOUNet, self).__init__()

        self.dims=[32, 64, 128]

        self.Encoder = nn.ModuleList([
            EBlock(self.dims[0], num_res=num_res),
            EBlock(self.dims[1], num_res=num_res),
            EBlock(self.dims[2], num_res=num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, self.dims[0], kernel_size=3, relu=True, stride=1),
            BasicConv(self.dims[0], self.dims[1], kernel_size=3, relu=True, stride=2),
            BasicConv(self.dims[1], self.dims[2], kernel_size=3, relu=True, stride=2),
            BasicConv(self.dims[2], self.dims[1], kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(self.dims[1], self.dims[0], kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(self.dims[0], 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(self.dims[2], num_res=num_res),
            DBlock(self.dims[1], num_res=num_res),
            DBlock(self.dims[0], num_res=num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(self.dims[2], self.dims[1], kernel_size=1, relu=True, stride=1),
            BasicConv(self.dims[1], self.dims[0], kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(self.dims[2], 3, kernel_size=3, relu=False, stride=1),
                BasicConv(self.dims[1], 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(self.dims[2])
        self.SCM1 = SCM(self.dims[2])
        self.FAM2 = FAM(self.dims[1])
        self.SCM2 = SCM(self.dims[1])

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        # 256
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        # 128
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        # 64
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        # 128
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        # 256
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)
        
        return outputs
    
    def flops(self, x):
        flops = 0
        _, H, W = x.shape
 
        z256 = torch.randn(1, self.dims[0], H, W, device='cuda')
        z128 = torch.randn(1, self.dims[1], H//2, W//2, device='cuda')
        z64 = torch.randn(1, self.dims[2], H//4, W//4, device='cuda')

        flops += self.Encoder[0].flops(z256)
        flops += self.Encoder[1].flops(z128)
        flops += self.Encoder[2].flops(z64)
        flops += self.Decoder[0].flops(z64)
        flops += self.Decoder[1].flops(z128)
        flops += self.Decoder[2].flops(z256)

        return flops

def build_net():
    return MIMOUNet()

