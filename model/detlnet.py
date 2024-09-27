import torch
import torch.nn as nn
import torch.nn.functional as F

from my_functionals import GatedSpatialConv as gsc
from network import Resnet
from torch.nn.parameter import Parameter
from dgcfm import *
from ciaam import *
from torch import Tensor
from collections import OrderedDict


class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


def autopad(k, p=None):
    # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        residual = x
        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        out = F.relu(x+residual, True)
        return out


class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)


class tfdblock(nn.Module):
    def __init__(self, inch, outch):
        super(tfdblock, self).__init__()
        self.res1 = Resnet.BasicBlock1(inch, outch, stride=1, downsample=None)   # cdc conv
        self.res2 = Resnet.BasicBlock1(inch, outch, stride=1, downsample=None)   # cdc conv
        self.gate = gsc.GatedSpatialConv2d(inch, outch)

    def forward(self,x,f_x):
        u_0 = x
        u_1, delta_u_0 = self.res1(u_0)
        _, u_2 = self.res2(u_1)

        u_3_pre = self.gate(u_2, f_x)
        u_3 = 3 * delta_u_0 + u_2 + u_3_pre
        return u_3


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            self.relu = nn.ReLU(inplace=True) if relu else None
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
            self.bn = None
            self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class detlnet(nn.Module):
    def __init__(self, layer_blocks, channels, n_blocks=7, base_dim=32):
        super(detlnet, self).__init__()

        stem_width = int(channels[0])

        self.stem = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, stem_width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),

            nn.Conv2d(stem_width, stem_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),

            nn.Conv2d(stem_width, 2*stem_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2*stem_width),
            nn.ReLU(True),

            nn.MaxPool2d(3, 2, 1),
        )
        self.ciaam_low = CIAAModule(channels[2])
        self.ciaam_high = CIAAModule(channels[1])

        self.layer1 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[0],
                                       in_channels=channels[1], out_channels=channels[1], stride=1)
        self.layer2 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[1],
                                       in_channels=channels[1], out_channels=channels[2], stride=2)
        self.layer3 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[2],
                                       in_channels=channels[2], out_channels=channels[3], stride=2)

        self.dgcfm1 = dgcfm(in_channels=channels[3], out_channels=channels[3])

        self.deconv2 = nn.ConvTranspose2d(channels[3], channels[2], 4, 2, 1)

        self.uplayer2 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[1],
                                         in_channels=channels[2], out_channels=channels[2], stride=1)
        self.dgcfm2 = dgcfm(in_channels=channels[2], out_channels=channels[2])

        self.deconv1 = nn.ConvTranspose2d(channels[2], channels[1], 4, 2, 1)

        self.uplayer1 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[0],
                                         in_channels=channels[1], out_channels=channels[1], stride=1)
        self.dgcfm3 = dgcfm(in_channels=channels[1], out_channels=channels[1])

        self.head = _FCNHead(channels[1], 1)

        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(32, 1, 1)
        self.dsn3 = nn.Conv2d(16, 1, 1)

        self.res1 = Resnet.BasicBlock(64, 64, stride=1, downsample=None)
        self.d1 = nn.Conv2d(64, 32, 1)

        self.res2 = Resnet.BasicBlock(32, 32, stride=1, downsample=None)
        self.d2 = nn.Conv2d(32, 16, 1)

        self.res3 = Resnet.BasicBlock(16, 16, stride=1, downsample=None)
        self.d3 = nn.Conv2d(16, 8, 1)
        self.fuse = nn.Conv2d(64, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(4, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)
        self.sigmoid = nn.Sigmoid()
        self.dsup = nn.Conv2d(3, 64, 1)
        self.head2 = _FCNHead(channels[1], 3)
        self.conv2_1 = nn.Conv2d(3, 1, 1)
        self.conv16 = nn.Conv2d(3, 16, 1)
        self.myb1 = tfdblock(64,64)
        self.myb2 = tfdblock(64,64)
        self.myb3 = tfdblock(64,64)

        self.CDC1 = CDC_conv(64, 32, kernel_size=3, padding=1, theta=0.7, padding_mode='zeros',
                             stride=1, bias=True)
        self.gate4 = gsc.GatedSpatialConv2d(64, 64)

    def forward(self, x, x_grad):
        _, _, hei, wid = x.shape
        x_size = x.size()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x1 = self.stem(x)
        c1 = self.layer1(x1)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)

        dgcfm1 = self.dgcfm1(c3)

        m1f = F.interpolate(x_grad, size=[hei, wid], mode='bilinear', align_corners=True)
        m1f = self.dsup(m1f)

        s1 = F.interpolate(self.dsn1(dgcfm1), size=[hei, wid], mode='bilinear', align_corners=True)
        cs1 = self.myb1(m1f, s1)

        dgcfm1_size = dgcfm1.size()
        cs1_1 = F.interpolate(cs1, size=dgcfm1_size[2:], mode='bilinear', align_corners=False)

        cs1_1 = self.sigmoid(cs1_1)
        et1 = cs1_1 * dgcfm1 + dgcfm1

        deconc2 = self.deconv2(et1)

        fusec2 = self.ciaam_low(deconc2, c2)
        upc2 = self.uplayer2(fusec2)

        upc2 = upc2 + c2
        dgcfm2 = self.dgcfm2(upc2)

        s2 = F.interpolate(self.dsn2(dgcfm2), size=[hei, wid], mode='bilinear', align_corners=True)

        cs2 = self.myb2(cs1, s2)

        dgcfm2_size = dgcfm2.size()
        cs2_2 = F.interpolate(cs2, size=dgcfm2_size[2:], mode='bilinear', align_corners=True)
        cs2_2 = self.CDC1(cs2_2)

        cs2_2 = self.sigmoid(cs2_2)
        et2 = cs2_2 * dgcfm2 + dgcfm2

        deconc1 = self.deconv1(et2)
        fusec1 =self.ciaam_high(deconc1, c1)
        upc1 = self.uplayer1(fusec1)
        upc1 = upc1 + c1
        dgcfm3 = self.dgcfm3(upc1)

        s3 = F.interpolate(self.dsn3(dgcfm3), size=[hei, wid], mode='bilinear', align_corners=True)
        cs = self.myb3(cs2, s3)

        cs = self.fuse(cs)
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)

        edge_out = self.sigmoid(cs)

        dgcfm3 = F.interpolate(dgcfm3, size=[hei, wid], mode='bilinear')
        fuse = edge_out * dgcfm3 + dgcfm3

        pred = self.head(fuse)

        out = F.interpolate(pred, size=[hei, wid], mode='bilinear')
        return out, edge_out

    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        layer = []
        downsample = (in_channels != out_channels) or (stride != 1)
        layer.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels, 1, False))
        return nn.Sequential(*layer)


if __name__ == '__main__':
    net = detlnet(layer_blocks = [4] * 3,
        channels = [8, 16, 32, 64])
    print(net)



