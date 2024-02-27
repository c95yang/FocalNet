import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
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
        )
        # self.scale = nn.Parameter(torch.ones(in_channel,1,1))
        # nn.init.normal_(self.scale, mean=1, std=.02)
        # self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        # self.conv2 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
    def forward(self, x):
        return self.main(x) + x
class ResBlock1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock1, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.main1 = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2):
        x1 = self.main(x1) + x1 
        x2 = self.main1(x2) + x2
        return x1, x2


class UNet(nn.Module):
    def __init__(self, inchannel, outchannel, num_res) -> None:
        super().__init__()



            # if  i < (num_res - num_res // 4):
        self.layers = ResBlock1(inchannel//2, outchannel//2) #ResBlock(16, 16)
            # else:
                # self.layers.append(ResBlock(inchannel, outchannel))
        self.num_res = num_res
        self.down = nn.Conv2d(inchannel//2, outchannel//2, kernel_size=2, stride=2, groups=inchannel//2)

    def forward(self, x):


        x1, x2 = torch.chunk(x, 2, dim=1)
        x2 = self.down(x2)
        # x1 = layer(x1)
        # x2 = layer(x2)
        x1, x2 = self.layers(x1, x2)
        x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear') #upsample
        x = torch.cat((x1, x2), dim=1) #torch.Size([4, 32, 256, 256])
            # elif 0 < i < (self.num_res - self.num_res // 4):
                # x1 = layer(x1)
                # x2 = layer(x2)
                # x1, x2 = layer(x1, x2)
            # elif i == self.num_res - 1:
            #     # i == self.num_res - 1:
            #     x1, x2 = layer(x1, x2)
            #     x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear')
            #     x = torch.cat((x1,x2), dim=1)
                
            # else:
            #     x1, x2 = layer(x1, x2)
        return x


    

if __name__ == '__main__':
    #model = UNet(32, 32, 8)
    x = torch.randn(4, 128, 256, 256)
    #y = model(x)
    #print(model)
    channel = 128
    #model = BasicConv1(channel, channel, 5, stride=1, dilation=2, padding=4, groups=channel)
    #model = BasicConv1(channel, channel, 7, stride=1, dilation=3, padding=9, groups=channel)
    #BasicConv1(channel, channel, kernel_size, stride=1, padding=1, groups=channel)
    #y = model(x)
    for i in range(5,-1,-1) :
        print(i)