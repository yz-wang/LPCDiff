import torch
import torch.nn as nn
from models.wavelet import DWT, IWT

dwt, iwt = DWT(), IWT()

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Dilated_Resblock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dilated_Resblock1, self).__init__()

        sequence = list()
        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1)),
            Swish(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1)),
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x) + x

        return out


class Dilated_Resblock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dilated_Resblock2, self).__init__()

        sequence = list()
        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),
            Swish(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x) + x

        return out


class Dilated_Resblock3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dilated_Resblock3, self).__init__()

        sequence = list()
        sequence += [

            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=3, dilation=(3, 3)),
            Swish(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=3, dilation=(3, 3))

        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x) + x

        return out


class Refine(nn.Module):
    def __init__(self, conv1=default_conv):
        super(Refine, self).__init__()
        self.dim = 64

        self.Dilated_Res1 = Dilated_Resblock1(self.dim, self.dim)
        self.Dilated_Res2 = Dilated_Resblock2(self.dim, self.dim)
        self.Dilated_Res3 = Dilated_Resblock3(self.dim, self.dim)

        kernel_size = 3

        pre_process = [conv1(3, self.dim, kernel_size=kernel_size)]

        post_precess = [
            conv1(self.dim * 3, self.dim, kernel_size),
            conv1(self.dim, 3, kernel_size=kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):

        n, c, h, w = x1.shape
        x1_dwt = dwt(x1)
        x1_LL, x1_high = x1_dwt[:n, ...], x1_dwt[n:, ...]
        res = x1_LL

        x1_LL = self.pre(x1_LL)
        x2 = self.Dilated_Res1(x1_LL)
        x3 = self.Dilated_Res2(x1_LL)
        x4 = self.Dilated_Res3(x1_LL)
        x = torch.cat((x2, x3, x4), dim=1)

        x1_LL = self.post(x)
        ll = res + x1_LL

        final = iwt(torch.cat((ll, x1_high), dim=0))

        return final