from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *
class BasicConv1(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, gelu=False, bn=False, bias=True):
        super(BasicConv1, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.gelu = nn.GELU() if gelu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.gelu is not None:
            x = self.gelu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 ) #output(n, 2, n, n)
class SpatialGate(nn.Module):
    def __init__(self, channel):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = BasicConv1(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, gelu=False)
        self.dw1 = nn.Sequential(
            BasicConv1(channel, channel, 5, stride=1, dilation=2, padding=4, groups=channel),
            BasicConv1(channel, channel, 7, stride=1, dilation=3, padding=9, groups=channel)
        )
        self.dw2 = BasicConv1(channel, channel, kernel_size, stride=1, padding=1, groups=channel)

    def forward(self, x):
        out = self.compress(x)
        out = self.spatial(out) # contains degradation locations to focus
        out = self.dw1(x) * out + self.dw2(x) #element-wise multiplication
        return out


class LocalAttention(nn.Module):
    def __init__(self, channel, p) -> None:
        super().__init__()
        self.channel = channel
        self.num_patch = 2 ** p
        self.sig = nn.Sigmoid()

        self.a = nn.Parameter(torch.zeros(channel,1,1)) 
        self.b = nn.Parameter(torch.ones(channel,1,1))

    def forward(self, x):
        out = x - torch.mean(x, dim=(2,3), keepdim=True) # high frequency information
        return self.a*out*x + self.b*x

class ParamidAttention(nn.Module):
    def __init__(self, channel) -> None: #x:([4, 128, 64, 64]) #channel = 128
        super().__init__()
        pyramid = 1
        self.spatial_gate = SpatialGate(channel)
        layers = [LocalAttention(channel, p=i) for i in range(pyramid-1,-1,-1)] #only one layer when p = 0?
        self.local_attention = nn.Sequential(*layers)
        self.a = nn.Parameter(torch.zeros(channel,1,1))
        self.b = nn.Parameter(torch.ones(channel,1,1))#torch.Size([128, 1, 1])
    def forward(self, x):
        out = self.spatial_gate(x) #torch.Size([4, 128, 64, 64])
        out = self.local_attention(out)
        return self.a*out + self.b*x #torch.Size([4, 128, 64, 64])

class EBlock_G(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock_G, self).__init__()
        layers = [ResBlock_G(out_channel, out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)
class DBlock_G(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock_G, self).__init__()
        layers = [ResBlock_G(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)
    
class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()
        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()
        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)
class EBlock1(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock1, self).__init__()
        self.layers = UNet(out_channel, out_channel, num_res)
    def forward(self, x):
        return self.layers(x)


class DBlock1(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock1, self).__init__()

        self.layers = UNet(channel, channel, num_res)
    def forward(self, x):
        return self.layers(x)

class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
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
    def __init__(self, num_res=1): #num_res=4 in focalnet
        super(MIMOUNet, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock1(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            #EBlock(base_channel*4, num_res),
            EBlock_G(base_channel*4, 8),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            #DBlock(base_channel * 4, num_res),
            DBlock_G(base_channel * 4, 8),
            DBlock(base_channel * 2, num_res),
            DBlock1(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        pyramid_attention = []
        for _ in range(4):
            pyramid_attention.append(ParamidAttention(base_channel * 4))
        self.pyramid_attentions = nn.Sequential(*pyramid_attention)
    def forward(self, x):
        #x: torch.Size([4, 3, 256, 256])
        x_2 = F.interpolate(x, scale_factor=0.5) #torch.Size([4, 3, 128, 128])
        x_4 = F.interpolate(x_2, scale_factor=0.5) #torch.Size([4, 3, 64, 64])
        z2 = self.SCM2(x_2) #torch.Size([4, 64, 128, 128])
        z4 = self.SCM1(x_4) #torch.Size([4, 128, 64, 64])

        outputs = list()
        # 256
        x_ = self.feat_extract[0](x) #torch.Size([4, 32, 256, 256])
        res1 = self.Encoder[0](x_) #torch.Size([4, 32, 256, 256]) #UNet(32, 32, 8)

        # 128
        z = self.feat_extract[1](res1) #torch.Size([4, 64, 128, 128])
        z = self.FAM2(z, z2)  #torch.Size([4, 64, 128, 128])
        res2 = self.Encoder[1](z)  #torch.Size([4, 64, 128, 128])

        # 64
        z = self.feat_extract[2](res2) #torch.Size([4, 128, 64, 64])
        z = self.FAM1(z, z4) #torch.Size([4, 128, 64, 64])
        z = self.Encoder[2](z) #torch.Size([4, 128, 64, 64]) #TODO: further reading!!
        z = self.pyramid_attentions(z)  #torch.Size([4, 128, 64, 64])   #TODO: further reading!!
        z = self.Decoder[0](z)  #torch.Size([4, 128, 64, 64])      #TODO: further reading!!
        z_ = self.ConvsOut[0](z) #torch.Size([4, 3, 64, 64])
        # 128
        z = self.feat_extract[3](z) #torch.Size([4, 64, 128, 128])
        outputs.append(z_+x_4) # print(outputs[0].shape) torch.Size([4, 3, 64, 64])

        z = torch.cat([z, res2], dim=1) #torch.Size([4, 128, 128, 128])
        z = self.Convs[0](z) #torch.Size([4, 64, 128, 128])
        z = self.Decoder[1](z) #torch.Size([4, 64, 128, 128])
        z_ = self.ConvsOut[1](z) #torch.Size([4, 3, 128, 128])

        # 256
        z = self.feat_extract[4](z) #torch.Size([4, 32, 256, 256])
        outputs.append(z_+x_2) #torch.Size([4, 3, 128, 128])

        z = torch.cat([z, res1], dim=1) #torch.Size([4, 64, 256, 256])
        z = self.Convs[1](z) #torch.Size([4, 32, 256, 256])
        z = self.Decoder[2](z) #torch.Size([4, 32, 256, 256])
        z = self.feat_extract[5](z) #torch.Size([4, 3, 256, 256])
        outputs.append(z+x) #torch.Size([4, 3, 256, 256])

        return outputs #3 outputs

class MIMOUNetPlus(nn.Module):
    def __init__(self, num_res = 20):
        super(MIMOUNetPlus, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        res2 = self.drop2(res2)
        res1 = self.drop1(res1)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


def build_net(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "MIMO-UNetPlus":
        return MIMOUNetPlus()
    elif model_name == "MIMO-UNet":
        return MIMOUNet()
    raise ModelError('Wrong Model!\nYou should choose MIMO-UNetPlus or MIMO-UNet.')
