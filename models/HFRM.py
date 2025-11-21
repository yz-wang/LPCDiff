import torch
import torch.nn as nn
from models.RDB import RDB
from models.coordatten import CoordinateAttention
import torch.nn.functional as F


class UNetConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_chans, out_chans, kernel_size=4, stride=2, padding=1))
        block.append(nn.ReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, up_mode):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
            )

        self.conv_block = nn.Sequential(
            nn.Conv2d(out_chans * 2, out_chans, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, bridge):
        # 上采样输入特征图 x
        up = self.up(x)

        diff_h = bridge.size(2) - up.size(2)
        diff_w = bridge.size(3) - up.size(3)

        up = F.pad(up, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2])

        # 拼接上采样后的特征图和跳跃连接特征图
        out = torch.cat([up, bridge], dim=1)

        # 通过卷积块处理拼接后的特征图
        out = self.conv_block(out)
        return out


class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class hfrm(nn.Module):
    def __init__(self, num_in_ch=3, base_channel=16, up_mode='upconv', bias=False):
        super(hfrm, self).__init__()
        assert up_mode in ('upconv', 'upsample')

        self.layer0 = Depth_conv(num_in_ch, base_channel)
        self.layer1 = UNetConvBlock(in_chans=16, out_chans=32)
        self.layer2 = UNetConvBlock(in_chans=32, out_chans=64)
        self.layer3 = UNetConvBlock(in_chans=64, out_chans=128)
        self.layer4 = UNetConvBlock(in_chans=128, out_chans=256)
        self.layer_0 = UNetUpBlock(in_chans=32, out_chans=16, up_mode=up_mode)
        self.layer_1 = UNetUpBlock(in_chans=64, out_chans=32, up_mode=up_mode)
        self.layer_2 = UNetUpBlock(in_chans=128, out_chans=64, up_mode=up_mode)
        self.layer_3 = UNetUpBlock(in_chans=256, out_chans=128, up_mode=up_mode)

        self.last = Depth_conv(base_channel, num_in_ch)

        self.RDB0 = RDB(nChannels=base_channel)  # 16
        self.RDB1 = RDB(nChannels=32)
        self.RDB2 = RDB(nChannels=64)
        self.RDB3 = RDB(nChannels=128)
        self.RDB4 = RDB(nChannels=256)
        self.RDB_0 = RDB(nChannels=base_channel)
        self.RDB_1 = RDB(nChannels=32)
        self.RDB_2 = RDB(nChannels=64)
        self.RDB_3 = RDB(nChannels=128)

        self.coordatten0 = CoordinateAttention(base_channel, base_channel)
        self.coordatten1 = CoordinateAttention(32, 32)
        self.coordatten2 = CoordinateAttention(64, 64)
        self.coordatten3 = CoordinateAttention(128, 128)

    def forward(self, x):

        blocks = []
        x = self.layer0(x)
        x0 = self.RDB0(x)
        blocks.append(self.coordatten0(x0))

        x1 = self.layer1(x0)
        x1 = self.RDB1(x1)
        blocks.append(self.coordatten1(x1))

        x2 = self.layer2(x1)
        x2 = self.RDB2(x2)
        blocks.append(self.coordatten2(x2))

        x3 = self.layer3(x2)
        x3 = self.RDB3(x3)
        blocks.append(self.coordatten3(x3))

        x4 = self.layer4(x3)
        x4 = self.RDB4(x4)

        blocks_up = [x4]
        x_3 = self.layer_3(x4, blocks[-0 - 1])
        x_3 = self.RDB_3(x_3)
        blocks_up.append(x_3)

        x_2 = self.layer_2(x_3, blocks[-1 - 1])
        x_2 = self.RDB_2(x_2)
        blocks_up.append(x_2)

        x_1 = self.layer_1(x_2, blocks[-2 - 1])
        x_1 = self.RDB_1(x_1)
        blocks_up.append(x_1)

        x_0 = self.layer_0(x_1, blocks[-3 - 1])
        x_0 = self.RDB_0(x_0)
        output = self.last(x_0)

        return output