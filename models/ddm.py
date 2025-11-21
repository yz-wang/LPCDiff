import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils
from models.unet import DiffusionUNet
from pytorch_msssim import ssim
from models.Lap import Lap_Pyramid_Conv
from models.HFRM import hfrm
from torch.optim.lr_scheduler import CosineAnnealingLR

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)



class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device

        self.Unet = DiffusionUNet(config)

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

        self.HFRM1 = hfrm()
        self.HFRM2 = hfrm()

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond, b, eta=0.):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)  # noise
        xs = [x]  # 迭代采样，存储采样结果的列表
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)  # 每个元素值都是i
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = self.Unet(torch.cat([x_cond, xt], dim=1), t)  # 预测的噪音

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()  # 预测clean,由x_t直接计算x_0

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et  # 下一个时间步的采样结果，迭代去噪
            xs.append(xt_next.to(x.device))

        return xs[-1]

    def forward(self, x):
        data_dict = {}
        lap = Lap_Pyramid_Conv(num_high=2)
        input_img = x[:, :3, :, :]  # hazy

        input_img_norm = data_transform(input_img)

        input_pyramid = lap.pyramid_decom(input_img_norm)

        high_1, high_2 = input_pyramid[0], input_pyramid[1]

        high_1, high_2 = self.HFRM1(high_1), self.HFRM2(high_2)

        input_LL = input_pyramid[-1]
        b = self.betas.to(input_img.device)

        t = torch.randint(low=0, high=self.num_timesteps, size=(input_LL.shape[0] // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:input_LL.shape[0]].to(x.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

        e = torch.randn_like(input_LL)


        # LPCDiff_k=2
        if self.training:
            gt_img_norm = data_transform(x[:, 3:, :, :])
            gt_pyramid = lap.pyramid_decom(gt_img_norm)
            gt_LL = gt_pyramid[-1]
            gt_high_1, gt_high_2 = gt_pyramid[0], gt_pyramid[1]

            x = gt_LL * a.sqrt() + e * (1.0 - a).sqrt()  # X为X_T， 在GT的基础上加噪音得到
            noise_output = self.Unet(torch.cat([input_LL, x], dim=1),
                                     t.float())  # Unet的输入为X_T和X_hazy的concat，在通道维度上对张量进行拼接，可以实现特征的融合和信息的整合

            pred_LL = self.sample_training(input_LL, b)  # 扩散模型预测的LL
            pred_x = lap.pyramid_recons([high_1, high_2, pred_LL])


            data_dict["gt"] = gt_img_norm
            data_dict["high_1"] = high_1
            data_dict["high_2"] = high_2
            data_dict["gt_high_1"] = gt_high_1
            data_dict["gt_high_2"] = gt_high_2
            data_dict["pred_LL"] = pred_LL
            data_dict["gt_LL"] = gt_LL
            data_dict["noise_output"] = noise_output
            data_dict["pred_x"] = pred_x
            data_dict["e"] = e

        else:
            pred_LL = self.sample_training(input_LL, b)
            pred_x = lap.pyramid_recons([high_1, high_2, pred_LL])
            pred_x = inverse_data_transform(pred_x)

            data_dict["pred_x"] = pred_x

        return data_dict


# class FFTLoss(nn.Module):
#     def __init__(self, loss_weight=1.0, reduction='mean'):
#         super(FFTLoss, self).__init__()
#         self.loss_weight = loss_weight
#         self.criterion = torch.nn.L1Loss(reduction=reduction)
#
#     def forward(self, pred, target):
#         pred_fft = torch.fft.rfft2(pred)
#         target_fft = torch.fft.rfft2(target)
#
#         pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
#         target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
#
#         return self.loss_weight * self.criterion(pred_fft, target_fft)


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # scheduler 单独创建
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=12000, eta_min=1e-6)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        # self.fft_loss = FFTLoss()

        self.start_epoch, self.step = 0, 0

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        # self.step = checkpoint['step']
        # self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        # self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        # print("=> loaded checkpoint {} step {}".format(load_path, self.step))

        # print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        try:
            if os.path.isfile(self.args.resume):
                self.load_ddm_ckpt(self.args.resume)
                print("导入扩散成功")
            else:
                print('Pre-trained diffusion model path is missing!')

        except Exception as e:
            print("导入扩散失败：", e)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):

            data_start = time.time()
            data_time = 0

            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                output = self.model(x)

                ssim_loss = 1 - ssim(output["pred_x"], output["gt"], data_range=1.0).to(self.device)
                high_loss = self.l1_loss(output["high_1"], output["gt_high_1"]) + self.l1_loss(output["high_2"], output["gt_high_2"])
                LL_loss = self.l1_loss(output["pred_LL"], output["gt_LL"])
                l1_loss = self.l1_loss(output["pred_x"], output["gt"])
                # fft_loss = self.fft_loss(output["pred_x"], gt)
                noise_loss = self.l2_loss(output["noise_output"], output["e"])
                loss = noise_loss + l1_loss + 0.2 * ssim_loss + LL_loss + high_loss
                # loss = noise_loss + l1_loss + ssim_loss + LL_loss + high_loss

                if self.step % 100 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch + 1}, lr={current_lr:.7f}")
                    print("epoch:{}, step:{},l1_loss:{:.4f}, ssim_loss:{:.4f}, noise_loss:{:.4f}, LL_loss:{:.4f}, high_loss:{:.4f}".format(epoch, self.step, l1_loss.item(), ssim_loss.item(), noise_loss.item(), LL_loss.item(), high_loss.item()))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)

                    utils.logging.save_checkpoint({
                                                   'state_dict': self.model.state_dict(),
                                                   # 'params': self.args,
                                                   # 'optimizer': self.optimizer.state_dict(),
                                                   # 'scheduler': self.scheduler.state_dict(),
                                                   # 'ema_helper': self.ema_helper.state_dict(),
                                                   # 'step': self.step, 'epoch': epoch + 1,
                                                   # 'config': self.config
                                                    },
                                                  filename=os.path.join(self.config.data.ckpt_dir, 'iter_{}'.format(self.step)))
            self.scheduler.step()


    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder, 'val_iter{}'.format(self.step))
        self.model.eval()
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):

                b, _, img_h, img_w = x.shape
                img_h_32 = int(32 * np.ceil(img_h / 32.0))
                img_w_32 = int(32 * np.ceil(img_w / 32.0))
                x = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')

                out = self.model(x.to(self.device))
                pred_x = out["pred_x"]
                pred_x = pred_x[:, :, :img_h, :img_w]
                utils.logging.save_image(pred_x, os.path.join(image_folder, f"{y[0].split('.')[0]}.png"))


