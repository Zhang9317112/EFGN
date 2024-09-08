import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# from torch import Module
# from torch import functional as F
#
# from torch import Tensor








def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):        # Conv2d input: [B，C，H，W]. W=((w-k+2p)//s)+1
    if dilation == 1:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size-1) // 2, bias=bias)
    elif dilation == 2:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=2, bias=bias, dilation=dilation)

    else:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=3, bias=bias, dilation=dilation)


# 这是扩张卷积
def default_conv2(in_channels, out_channels, kernel_size, stride=1, bias=True, dilation=1):
    if dilation==1:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=(kernel_size//2), bias=bias)
    elif dilation==2:
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=2, bias=bias, dilation=dilation)
    else:
       padding = int((kernel_size - 1) / 2) * dilation
       return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           stride, padding=padding, bias=bias, dilation=dilation)


def prosessing_conv(in_channels, out_channels, kernel_size, stride, bias=True):      # W=((w-k+2p)//s)+1. [C,H,W]->[C,H/s,W/s]: k-2p=s.
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=bias)
    # s=2,h=w=8;s=3,h=w=6

def transpose_conv(in_channels, out_channels, kernel_size, stride, bias=True):       # [C,H/s,W/s]->[C,H,W]
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=1, output_padding=1, bias=bias)

    # output = (input-1)*stride + outputpadding - 2*padding + kernelsize
    # 2p-op=k-s
    # s=2,p=1,outp=1,h=w=16, 2*pading-outpadding=1
    # s=3,p=1,outp=0,h=w=16, 2*pading-outpadding=2



class CALayer(nn.Module):               # channel attention mechanism
    def __init__(self, in_channels, reduction_rate=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_rate, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_rate, in_channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class Upsampler(nn.Sequential):
    def __init__(self, conv, up_scale, in_feats, bn=False, act=False, bias=True):
        m = []
        if (up_scale & (up_scale - 1)) == 0:
            for _ in range(int(math.log(up_scale, 2))):
                m.append(conv(in_feats, 4 * in_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(in_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(in_feats))

        elif up_scale == 3:
            m.append(conv(in_feats, 9 * in_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(in_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(in_feats))


        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)