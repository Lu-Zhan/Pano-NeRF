from typing import Tuple
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.surface_rendering import solid_angle_refinement

# all copyright belong to pytorch geometry
def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2) ** 2 / float(2 * sigma ** 2)

    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()


def get_gaussian_kernel(ksize: int, sigma: float) -> torch.Tensor:
    if not isinstance(ksize, int) or ksize % 2 == 0 or ksize <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(ksize))
    window_1d: torch.Tensor = gaussian(ksize, sigma)
    return window_1d


def get_gaussian_kernel2d(ksize: Tuple[int, int],
                          sigma: Tuple[float, float]) -> torch.Tensor:
    if not isinstance(ksize, tuple) or len(ksize) != 2:
        raise TypeError("ksize must be a tuple of length two. Got {}"
                        .format(ksize))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}"
                        .format(sigma))
    ksize_x, ksize_y = ksize
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y: torch.Tensor = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d


class SSIM(nn.Module):
    def __init__(
            self,
            window_size: int,
            reduction: str = 'none',
            max_val: float = 1.0) -> None:
        super(SSIM, self).__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.reduction: str = reduction

        self.window: torch.Tensor = get_gaussian_kernel2d(
            (window_size, window_size), (1.5, 1.5))
        self.padding: int = self.compute_zero_padding(window_size)

        self.C1: float = (0.01 * self.max_val) ** 2
        self.C2: float = (0.03 * self.max_val) ** 2

    @staticmethod
    def compute_zero_padding(kernel_size: int) -> int:
        """Computes zero padding."""
        return (kernel_size - 1) // 2

    def filter2D(
            self,
            input: torch.Tensor,
            kernel: torch.Tensor,
            channel: int) -> torch.Tensor:
        return F.conv2d(input, kernel, padding=self.padding, groups=channel)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(img1):
            raise TypeError("Input img1 type is not a torch.Tensor. Got {}"
                            .format(type(img1)))
        if not torch.is_tensor(img2):
            raise TypeError("Input img2 type is not a torch.Tensor. Got {}"
                            .format(type(img2)))
        if not len(img1.shape) == 4:
            raise ValueError("Invalid img1 shape, we expect BxCxHxW. Got: {}"
                             .format(img1.shape))
        if not len(img2.shape) == 4:
            raise ValueError("Invalid img2 shape, we expect BxCxHxW. Got: {}"
                             .format(img2.shape))
        if not img1.shape == img2.shape:
            raise ValueError("img1 and img2 shapes must be the same. Got: {}"
                             .format(img1.shape, img2.shape))
        if not img1.device == img2.device:
            raise ValueError("img1 and img2 must be in the same device. Got: {}"
                             .format(img1.device, img2.device))
        if not img1.dtype == img2.dtype:
            raise ValueError("img1 and img2 must be in the same dtype. Got: {}"
                             .format(img1.dtype, img2.dtype))
        # prepare kernel
        b, c, h, w = img1.shape
        tmp_kernel: torch.Tensor = self.window.to(img1.device).to(img1.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        # compute local mean per channel
        mu1: torch.Tensor = self.filter2D(img1, kernel, c)
        mu2: torch.Tensor = self.filter2D(img2, kernel, c)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # compute local sigma per channel
        sigma1_sq = self.filter2D(img1 * img1, kernel, c) - mu1_sq
        sigma2_sq = self.filter2D(img2 * img2, kernel, c) - mu2_sq
        sigma12 = self.filter2D(img1 * img2, kernel, c) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        loss = ssim_map
        # loss = torch.clamp(1. - ssim_map, min=0, max=1) / 2.

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            pass
        return loss


def summarize(folder, num_buckets):
    metric_names = ['psnrs', 'ssims']
    precisions = [4, 4, 4, 4]

    results = []
    values = []
    for metric_name in metric_names:
        filename = os.path.join(folder, f'{metric_name}.txt')
        with open(filename) as f:
            v = np.array([float(s) for s in f.readline().split(' ')])
            values.append(np.mean(np.reshape(v, [-1, num_buckets]), 0))
    results.append(np.concatenate(values))
    avg_results = np.mean(np.array(results), 0)

    # psnr, ssim, lpips = np.mean(np.reshape(avg_results, [-1, num_buckets]), 1)
    psnr, ssim = np.mean(np.reshape(avg_results, [-1, num_buckets]), 1)

    mse = np.exp(-0.1 * np.log(10.) * psnr)
    dssim = np.sqrt(1 - ssim)
    avg_avg = np.exp(np.mean(np.log(np.array([mse, dssim]))))

    s = []
    for i, v in enumerate(np.reshape(avg_results, [-1, num_buckets])):
        s.append(' '.join([f'{s:0.{precisions[i]}f}' for s in v]))
    s.append(f'{avg_avg:0.{precisions[-1]}f}')
    return ' | '.join(s)


def summarize_results(folder, scene_names, num_buckets):
    metric_names = ['psnrs', 'ssims']
    precisions = [4, 4, 4, 4]

    results = []
    for scene_name in scene_names:
        values = []
        for metric_name in metric_names:
            filename = os.path.join(folder, 'test', scene_name, f'{metric_name}.txt')
            with open(filename) as f:
                v = np.array([float(s) for s in f.readline().split(' ')])
                values.append(np.mean(np.reshape(v, [-1, num_buckets]), 0))
        results.append(np.concatenate(values))
    avg_results = np.mean(np.array(results), 0)

    # psnr, ssim, lpips = np.mean(np.reshape(avg_results, [-1, num_buckets]), 1)
    psnr, ssim = np.mean(np.reshape(avg_results, [-1, num_buckets]), 1)

    mse = np.exp(-0.1 * np.log(10.) * psnr)
    dssim = np.sqrt(1 - ssim)
    avg_avg = np.exp(np.mean(np.log(np.array([mse, dssim]))))

    s = []
    for i, v in enumerate(np.reshape(avg_results, [-1, num_buckets])):
        s.append(' '.join([f'{s:0.{precisions[i]}f}' for s in v]))
    s.append(f'{avg_avg:0.{precisions[-1]}f}')
    return ' | '.join(s)


def ssim(
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11,
        reduction: str = 'none',
        max_val: float = 1.0) -> torch.Tensor:
    r"""Function that measures the Structural Similarity (SSIM) index between
    each element in the input `x` and target `y`.

    See :class:`torchgeometry.losses.SSIM` for details.
        - Input: :math:`(B, C, H, W)`
        - Target :math:`(B, C, H, W)`
        - Output: scale, if reduction is 'none', then :math:`(B, C, H, W)`
    """
    return SSIM(window_size, reduction, max_val)(img1, img2)

def calc_ssim(x: torch.Tensor, y: torch.Tensor):
    """
    Calculate SSIM between x and y
    """
    assert x.shape[0] == 1 or x.shape[0] == 3, 'x must be [B C H W]'
    return ssim(x, y, window_size=11, reduction='mean')


def calc_mse(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates the mean square error between tensors `x` and `y`.
    """
    return torch.mean((x - y) ** 2)


def calc_rmse(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates root mean square error between tensors `x` and `y`.
    """
    return torch.mean((x - y) ** 2) ** 0.5


def calc_l1(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates the mean abs error between tensors `x` and `y`.
    """
    return torch.abs(x - y).mean()


def calc_psnr(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`.
    """
    mse = calc_mse(x, y)
    psnr = -10.0 * torch.log10(mse)
    return psnr


def calc_mae(x: torch.Tensor, y: torch.Tensor, dim: int = -1):
    """
    Calculates the mean angle error between tensors `x` and `y`.
    """
    if dim == 1:
        x = x.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
    cos = F.cosine_similarity(x.reshape(-1, 3), y.reshape(-1, 3), dim=-1)
    angle = torch.acos(cos) / np.pi * 180
    angle = torch.nan_to_num(angle, nan=0.0)
    return angle.mean()


def calc_cossimi(x: torch.Tensor, y: torch.Tensor, dim: int = -1):
    """
    Calculates the cosine similarity between tensors `x` and `y`.
    """
    return F.cosine_similarity(x, y, dim=dim).mean()


def calc_lpips(x: torch.Tensor, y: torch.Tensor, loss_fn=None):
    """
    Calculates the LPIPS metric between tensors `x` and `y`. x and y should be within [-1, 1]
    """
    if loss_fn is None:
        import lpips
        loss_fn = lpips.LPIPS(net='vgg')
    loss_fn.to(x.device)
    return loss_fn(x * 2 - 1, y * 2 - 1)


def calc_relative_error(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates the relative error between tensors `x` and `y`.
    """
    scale = torch.linalg.lstsq(x, y).solution[0][0]
    return (torch.abs(y - x * scale) / (y + 1e-8)).mean()


def eval_errors(pred_color: torch.Tensor, batch_pixels: torch.Tensor):
    psnr_val = calc_psnr(pred_color, batch_pixels)
    if pred_color.shape[-1] == 3 and batch_pixels.shape[-1] == 3:
        pred_color = pred_color.permute(0, 3, 1, 2)
        batch_pixels = batch_pixels.permute(0, 3, 1, 2)
    ssim_val = ssim(pred_color, batch_pixels, window_size=11, reduction='mean')
    return psnr_val, ssim_val


"""Depth estimation metrics."""

def abs_rel_error(pred, gt, mask):
    '''Compute absolute relative difference error'''
    return ((pred[mask>0] - gt[mask>0]).abs() / gt[mask>0]).mean()


def sq_rel_error(pred, gt, mask):
    '''Compute squared relative difference error'''
    return (((pred[mask>0] - gt[mask>0]) ** 2) / gt[mask>0]).mean()


def lin_rms_sq_error(pred, gt, mask):
    '''Compute the linear RMS error except the final square-root step'''
    return torch.sqrt(((pred[mask>0] - gt[mask>0]) ** 2).mean())


def log_rms_sq_error(pred, gt, mask):
    '''Compute the log RMS error except the final square-root step'''
    mask = (mask > 0) & (pred > 1e-7) & (gt > 1e-7) # Compute a mask of valid values
    return torch.sqrt(((pred[mask].log() - gt[mask].log()) ** 2).mean())


def delta_inlier_ratio(pred, gt, mask, degree=1):
    '''Compute the delta inlier rate to a specified degree (def: 1)'''
    return (torch.max(pred[mask>0] / gt[mask>0], gt[mask>0] / pred[mask>0]) < (1.25 ** degree)).float().mean()


"""HDR estimation metrics."""

def calc_ws_psnr(pred, gt):
    '''Compute the weighted PSNR'''
    c, h, w = pred.shape
    weights = solid_angle_refinement(h=h, w=w).reshape(1, h, w).to(pred.device)
    weights = weights / weights.sum()

    mse = torch.sum((pred - gt) ** 2 * weights)
    psnr = -10.0 * torch.log10(mse)
    return psnr


def calc_ws_l1(pred, gt):
    '''Compute the weighted l1(mean absolute error)'''
    c, h, w = pred.shape
    weights = solid_angle_refinement(h=h, w=w).reshape(1, h, w).to(pred.device)
    weights = weights / weights.sum()

    mae = torch.sum(torch.abs(pred - gt) * weights)
    return mae


def calc_ws_mse(pred, gt):
    '''Compute the weighted MSE'''
    c, h, w = pred.shape
    weights = solid_angle_refinement(h=h, w=w).reshape(1, h, w).to(pred.device)
    weights = weights / weights.sum()

    rmse = torch.sum((pred - gt) ** 2 * weights)
    return rmse


def calc_ws_rmse(pred, gt):
    '''Compute the weighted RMSE'''
    c, h, w = pred.shape
    weights = solid_angle_refinement(h=h, w=w).reshape(1, h, w).to(pred.device)
    weights = weights / weights.sum()

    rmse = torch.sqrt(torch.sum((pred - gt) ** 2 * weights))
    return rmse


def calc_ws_mae(x: torch.Tensor, y: torch.Tensor, dim: int = -1, weights=None):
    """
    Calculates the mean angle error between tensors `x` and `y`.
    """
    if dim == 1:
        x = x.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)

    if weights is None:
        b, h, w, c = x.shape
        weights = solid_angle_refinement(h=h, w=w).reshape(-1).to(x.device)
        weights = weights / weights.sum()   #
    else:
        weights = weights.reshape(-1).to(x.device)
        weights = weights / weights.sum()

    cos = F.cosine_similarity(x.reshape(-1, 3), y.reshape(-1, 3), dim=-1)   # [n]
    angle = torch.acos(cos) / np.pi * 180
    angle = torch.nan_to_num(angle, nan=0)

    return torch.sum(angle * weights)


def calc_ws_cossimi(x: torch.Tensor, y: torch.Tensor, dim: int = 0):
    '''Compute the weighted similarity'''
    if dim == 0:
        c, h, w = x.shape
    elif dim == -1:
        h, w, c = x.shape
    elif dim == 1:
        n, c, h, w = x.shape

    weights = solid_angle_refinement(h=h, w=w).reshape(1, h, w).to(x.device)
    weights = weights / weights.sum()

    cos = F.cosine_similarity(x, y, dim=dim).reshape(1, h, w)

    return torch.sum(cos * weights)


"""Albedo estimation metrics."""
def calc_simse(x: torch.Tensor, y: torch.Tensor):
    """
    Calculates the scale-invariant mean square error between tensors `x` and `y`.
    """
    return torch.var(x - y)


if __name__ == '__main__':
    # from torch.nn.functional import normalize
    #
    # x = torch.randn(1, 3, 224, 224) * 2 - 1
    # y = torch.randn(1, 3, 224, 224) * 2 - 1
    #
    # x = normalize(x, dim=1)
    # y = normalize(y, dim=1)
    #
    # print(calc_mae(x, y, dim=1))
    #
    # cos = torch.sum(x * y, dim=1).mean()
    # angle = torch.acos(cos) / np.pi * 180
    # print(angle)
    #
    # cos = torch.sum(x * y, dim=1)
    # angle = torch.acos(cos) / np.pi * 180
    # print(angle.mean())

    a = torch.ones(1, 1) * 0.9
    b = torch.ones(1, 1) * 0.6

    print(calc_mse(a, b), -10*torch.log10(calc_mse(a, b)), calc_psnr(a, b))
