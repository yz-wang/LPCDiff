import torch
import torch.nn as nn
import cv2
import utils
def Normalize(x):
    ymax = 255
    ymin = 0
    xmax = x.max()
    xmin = x.min()
    return (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin


def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):  # 逆变换
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


if __name__ == "__main__":
    # x = torch.randn(4, 3, 256, 256)
    dwt = DWT()
    iwt = IWT()  # 不实例化直接调用会报错
    # LL = dwt(x)[:4, ...]
    # LL_LL = dwt(LL)[:4, ...]
    # print(LL.size())
    # print(LL_LL.size())
    img = cv2.imread('../visual_wavelet/image/1445.png')
    hazy = cv2.imread('../visual_wavelet/image/1445_5_pred.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hazy = cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img.transpose((2, 0, 1))).float().unsqueeze(0) / 255.0
    hazy_tensor = torch.tensor(hazy.transpose((2, 0, 1))).float().unsqueeze(0) / 255.0
    input_dwt = dwt(img_tensor)
    hazy_dwt = dwt(hazy_tensor)
    input_LL, input_high0 = input_dwt[:1, ...], input_dwt[1:, ...]
    hazy_LL, hazy_high0 = hazy_dwt[:1, ...], hazy_dwt[1:, ...]
    img_iwt = iwt(torch.cat((input_LL, hazy_high0)))  # cat里边不加括号会报错
    # utils.logging.save_image(input_high0[2:, ...], '../visual_wavelet/image/1445_HH.png')
    # utils.logging.save_image(hazy_high0[2:, ...], '../visual_wavelet/image/1445_5_pred_HH.png')
    utils.logging.save_image(img_iwt, '../visual_wavelet/image/1455_gt_LL_and_pred_h.png')




