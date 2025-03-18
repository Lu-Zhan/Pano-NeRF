import torch
from einops import rearrange, repeat
import numpy as np
from datasets.base_datasets import Rays_keys, Rays
from torch.func import jacrev, vmap


def lift_gaussian(directions, t_mean, t_var, r_var, diagonal):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    mean = torch.unsqueeze(directions, dim=-2) * torch.unsqueeze(t_mean, dim=-1)  # [B, 1, 3]*[B, N, 1] = [B, N, 3]
    d_norm_denominator = torch.sum(directions ** 2, dim=-1, keepdim=True) + 1e-10
    # min_denominator = torch.full_like(d_norm_denominator, 1e-10)
    # d_norm_denominator = torch.maximum(min_denominator, d_norm_denominator)

    if diagonal:
        d_outer_diag = directions ** 2  # eq (16)
        null_outer_diag = 1 - d_outer_diag / d_norm_denominator
        t_cov_diag = torch.unsqueeze(t_var, dim=-1) * torch.unsqueeze(d_outer_diag,
                                                                      dim=-2)  # [B, N, 1] * [B, 1, 3] = [B, N, 3]
        xy_cov_diag = torch.unsqueeze(r_var, dim=-1) * torch.unsqueeze(null_outer_diag, dim=-2)
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = torch.unsqueeze(directions, dim=-1) * torch.unsqueeze(directions,
                                                                        dim=-2)  # [B, 3, 1] * [B, 1, 3] = [B, 3, 3]
        eye = torch.eye(directions.shape[-1], device=directions.device)  # [B, 3, 3]
        # [B, 3, 1] * ([B, 3] / [B, 1])[..., None, :] = [B, 3, 3]
        null_outer = eye - torch.unsqueeze(directions, dim=-1) * (directions / d_norm_denominator).unsqueeze(-2)
        t_cov = t_var.unsqueeze(-1).unsqueeze(-1) * d_outer.unsqueeze(-3)  # [B, N, 1, 1] * [B, 1, 3, 3] = [B, N, 3, 3]
        xy_cov = t_var.unsqueeze(-1).unsqueeze(-1) * null_outer.unsqueeze(
            -3)  # [B, N, 1, 1] * [B, 1, 3, 3] = [B, N, 3, 3]
        cov = t_cov + xy_cov
        return mean, cov


def conical_frustum_to_gaussian(directions, t0, t1, base_radius, diagonal, stable=True):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).
    Assumes the ray is originating from the origin, and base_radius is the
    radius at dist=1. Doesn't assume `directions` is normalized.
    Args:
        directions: torch.tensor float32 3-vector, the axis of the cone
        t0: float, the starting distance of the frustum.
        t1: float, the ending distance of the frustum.
        base_radius: float, the scale of the radius as a function of distance.
        diagonal: boolean, whether or the Gaussian will be diagonal or full-covariance.
        stable: boolean, whether or not to use the stable computation described in
        the paper (setting this to False will cause catastrophic failure).
    Returns:
        a Gaussian (mean and covariance).
    """
    if stable:
        mu = (t0 + t1) / 2
        hw = (t1 - t0) / 2
        t_mean = mu + (2 * mu * hw ** 2) / (3 * mu ** 2 + hw ** 2)
        t_var = (hw ** 2) / 3 - (4 / 15) * ((hw ** 4 * (12 * mu ** 2 - hw ** 2)) /
                                            (3 * mu ** 2 + hw ** 2) ** 2)
        r_var = base_radius ** 2 * ((mu ** 2) / 4 + (5 / 12) * hw ** 2 - 4 / 15 *
                                    (hw ** 4) / (3 * mu ** 2 + hw ** 2))
    else:
        t_mean = (3 * (t1 ** 4 - t0 ** 4)) / (4 * (t1 ** 3 - t0 ** 3))
        r_var = base_radius ** 2 * (3 / 20 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3))
        t_mosq = 3 / 5 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3)
        t_var = t_mosq - t_mean ** 2
    return lift_gaussian(directions, t_mean, t_var, r_var, diagonal)


def cast_rays(t_samples, origins, directions, radii, ray_shape, diagonal=True):
    """Cast rays (cone- or cylinder-shaped) and featurize sections of it.
    Args:
        t_samples: float array [B, n_sample+1], the "fencepost" distances along the ray.
        origins: float array [B, 3], the ray origin coordinates.
        directions [B, 3]: float array, the ray direction vectors.
        radii[B, 1]: float array, the radii (base radii for cones) of the rays.
        ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
        diagonal: boolean, whether or not the covariance matrices should be diagonal.
    Returns:
        a tuple of arrays of means and covariances.
    """
    t0 = t_samples[..., :-1]  # [B, n_samples]
    t1 = t_samples[..., 1:]
    if ray_shape == 'cone':
        gaussian_fn = conical_frustum_to_gaussian
    elif ray_shape == 'cylinder':
        raise NotImplementedError
    else:
        assert False
    means, covs = gaussian_fn(directions, t0, t1, radii, diagonal)
    means = means + torch.unsqueeze(origins, dim=-2)
    return means, covs


def sample_along_rays_360(origins, directions, radii, num_samples, near, far, randomized, disparity, ray_shape):
    batch_size = origins.shape[0]
    t_samples = torch.linspace(0., 1., num_samples + 1, device=origins.device)
    far_inv = 1 / far
    near_inv = 1 / near
    t_inv = far_inv * t_samples + (1 - t_samples) * near_inv

    if randomized:
        mids = 0.5 * (t_inv[..., 1:] + t_inv[..., :-1])
        upper = torch.cat([mids, t_inv[..., -1:]], -1)
        lower = torch.cat([t_inv[..., :1], mids], -1)
        t_rand = torch.rand(batch_size, num_samples + 1, device=origins.device)
        t_inv = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_inv to make the returned shape consistent.
        t_inv = torch.broadcast_to(t_inv, [batch_size, num_samples + 1])
    t = 1 / t_inv
    means, covs = cast_rays(t, origins, directions, radii, ray_shape, False)
    return t_inv, (means, covs)


def sample_along_rays(origins, directions, radii, num_samples, near, far, randomized, disparity, ray_shape):
    """
    Stratified sampling along the rays.
    Args:
        origins: torch.Tensor, [batch_size, 3], ray origins.
        directions: torch.Tensor, [batch_size, 3], ray directions.
        radii: torch.Tensor, [batch_size, 3], ray radii.
        num_samples: int.
        near: torch.Tensor, [batch_size, 1], near clip.
        far: torch.Tensor, [batch_size, 1], far clip.
        randomized: bool, use randomized stratified sampling.
        disparity: bool, sampling linearly in disparity rather than depth.
        ray_shape: string, which shape ray to assume.
    Returns:
    t_samples: torch.Tensor, [batch_size, num_samples], sampled z values.
    means: torch.Tensor, [batch_size, num_samples, 3], sampled means.
    covs: torch.Tensor, [batch_size, num_samples, 3, 3], sampled covariances.
    """
    batch_size = origins.shape[0]

    t_samples = torch.linspace(0., 1., num_samples + 1, device=origins.device)

    if disparity:
        t_samples = 1. / (1. / near * (1. - t_samples) + 1. / far * t_samples)
    else:
        # t_samples = near * (1. - t_samples) + far * t_samples
        t_samples = near + (far - near) * t_samples

    if randomized:
        mids = 0.5 * (t_samples[..., 1:] + t_samples[..., :-1])
        upper = torch.cat([mids, t_samples[..., -1:]], -1)
        lower = torch.cat([t_samples[..., :1], mids], -1)
        t_rand = torch.rand(batch_size, num_samples + 1, device=origins.device)
        t_samples = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_samples to make the returned shape consistent.
        t_samples = torch.broadcast_to(t_samples, [batch_size, num_samples + 1])
    means, covs = cast_rays(t_samples, origins, directions, radii, ray_shape)
    return t_samples, (means, covs)


def sample_each_points(point_origins, directions, num_samples, near, far, radii, randomized):
    """
    Stratified sampling along the rays.
    Args:
        point_origins: torch.Tensor, [batch_size, num_points, 3], ray origins.
        directions: torch.Tensor, [num_lit_rays, 3], ray directions.
        num_samples: int.
        near: torch.Tensor, [num_lit_rays, 1], near clip.
        far: torch.Tensor, [num_lit_rays, 1], far clip.
        radii: torch.Tensor, [num_lit_rays, 3], ray radii.
        randomized: bool, use randomized stratified sampling.
    Returns:
    t_samples: torch.Tensor, [batch_size, num_samples + 1], sampled z values.
    means: torch.Tensor, [batch_size, num_samples, 3], sampled means.
    covs: torch.Tensor, [batch_size, num_samples, 3, 3], sampled covariances.
    """
    bz, num_points, _ = point_origins.shape
    num_dirs = directions.shape[0]
    origins = repeat(point_origins, 'b np c -> b np dir c', dir=num_dirs).reshape(-1, 3)
    directions = repeat(directions, 'dir c -> b np dir c', b=bz, np=num_points).reshape(-1, 3)
    radii = repeat(radii, 'dir c -> b np dir c', b=bz, np=num_points).reshape(-1, 1)
    near = repeat(near, 'dir c -> b np dir c', b=bz, np=num_points).reshape(-1, 1)
    far = repeat(far, 'dir c -> b np dir c', b=bz, np=num_points).reshape(-1, 1)

    batch_size = origins.shape[0]   # bz * num_points * num_lit_rays
    t_samples = torch.linspace(0., 1., num_samples + 1, device=origins.device)
    t_samples = near + (far - near) * t_samples # [bz * np * dir, num_samples + 1]

    if randomized:
        mids = 0.5 * (t_samples[..., 1:] + t_samples[..., :-1])
        upper = torch.cat([mids, t_samples[..., -1:]], -1)
        lower = torch.cat([t_samples[..., :1], mids], -1)
        t_rand = torch.rand(num_points, num_samples + 1, device=origins.device)
        t_samples = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_samples to make the returned shape consistent.
        t_samples = torch.broadcast_to(t_samples, [batch_size, num_samples + 1])

    means, covs = cast_rays(t_samples, origins, directions, radii, 'cone')  # [batch_size, num_samples, 3]

    return t_samples, (means, covs), directions


def sample_each_points_hemisp(point_origins, directions, num_samples, near, far, radii, randomized):
    """
    Stratified sampling along the rays (follow normal on the upper sphere).
    Args:
        point_origins: torch.Tensor, [batch_size, num_points, 3], ray origins.
        directions: torch.Tensor, [batch_size, num_lit_rays, 3], ray directions.
        num_samples: int.
        near: torch.Tensor, [num_lit_rays, 1], near clip.
        far: torch.Tensor, [num_lit_rays, 1], far clip.
        radii: torch.Tensor, [num_lit_rays, 3], ray radii.
        randomized: bool, use randomized stratified sampling.
    Returns:
    t_samples: torch.Tensor, [batch_size, num_samples], sampled z values.
    means: torch.Tensor, [batch_size, num_samples, 3], sampled means.
    covs: torch.Tensor, [batch_size, num_samples, 3, 3], sampled covariances.
    """
    bz, num_points, _ = point_origins.shape
    num_dirs = directions.shape[1]
    origins = repeat(point_origins, 'b np c -> b np dir c', dir=num_dirs).reshape(-1, 3)
    directions = repeat(directions, 'b dir c -> b np dir c', np=num_points).reshape(-1, 3)
    radii = repeat(radii, 'dir c -> b np dir c', b=bz, np=num_points).reshape(-1, 1)
    near = repeat(near, 'dir c -> b np dir c', b=bz, np=num_points).reshape(-1, 1)
    far = repeat(far, 'dir c -> b np dir c', b=bz, np=num_points).reshape(-1, 1)

    batch_size = origins.shape[0]   # bz * num_points * num_lit_rays
    t_samples = torch.linspace(0., 1., num_samples + 1, device=origins.device)
    t_samples = near + (far - near) * t_samples # [bz * np * dir, num_samples + 1]

    if randomized:
        mids = 0.5 * (t_samples[..., 1:] + t_samples[..., :-1])
        upper = torch.cat([mids, t_samples[..., -1:]], -1)
        lower = torch.cat([t_samples[..., :1], mids], -1)
        t_rand = torch.rand(num_points, num_samples + 1, device=origins.device)
        t_samples = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_samples to make the returned shape consistent.
        t_samples = torch.broadcast_to(t_samples, [batch_size, num_samples + 1])

    means, covs = cast_rays(t_samples, origins, directions, radii, 'cone')  # [batch_size, num_samples, 3]

    return t_samples, (means, covs), directions


def sorted_piecewise_constant_pdf(bins, weights, num_samples, randomized):
    """
    Piecewise-Constant PDF sampling from sorted bins.
    Args:
        bins: torch.Tensor, [batch_size, num_bins + 1].
        weights: torch.Tensor, [batch_size, num_bins].
        num_samples: int, the number of samples.
        randomized: bool, use randomized samples.
    Returns:
        t_samples: torch.Tensor, [batch_size, num_samples].
    """
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    eps = 1e-5
    weight_sum = torch.sum(weights, dim=-1, keepdim=True)  # [B, 1]
    padding = torch.maximum(torch.zeros_like(weight_sum), eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
    # starts with exactly 0 and ends with exactly 1.
    pdf = weights / weight_sum
    cdf = torch.cumsum(pdf[..., :-1], dim=-1)
    cdf = torch.minimum(torch.ones_like(cdf), cdf)
    cdf = torch.cat([torch.zeros(list(cdf.shape[:-1]) + [1], device=cdf.device),
                     cdf,
                     torch.ones(list(cdf.shape[:-1]) + [1], device=cdf.device)],
                    dim=-1)  # [B, N]

    # Draw uniform samples.
    if randomized:
        s = 1 / num_samples
        u = (torch.arange(num_samples, device=cdf.device) * s)[None, ...]
        u = u + torch.empty(list(cdf.shape[:-1]) + [num_samples], device=cdf.device).uniform_(
            to=(s - torch.finfo(torch.float32).eps))
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.minimum(u, torch.full_like(u, 1. - torch.finfo(torch.float32).eps, device=u.device))
    else:
        # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
        u = torch.linspace(0., 1. - torch.finfo(torch.float32).eps, num_samples, device=cdf.device)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])
    u = u.contiguous()
    try:
        inds = torch.searchsorted(cdf, u, right=True)
    except:
        # for lower version torch that does not have torch.searchsorted,
        # you need to manually install from
        # https://github.com/aliutkus/torchsearchsorted
        from torchsearchsorted import searchsorted
        inds = searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples


def resample_along_rays(origins, directions, radii, t_samples, weights, randomized, ray_shape, stop_grad,
                        resample_padding):
    """Resampling.
    Args:
        origins: torch.Tensor, [batch_size, 3], ray origins.
        directions: torch.Tensor, [batch_size, 3], ray directions.
        radii: torch.Tensor, [batch_size, 3], ray radii.
        t_samples: torch.Tensor, [batch_size, num_samples+1].
        weights: torch.Tensor [batch_size, num_samples], weights for t_samples
        randomized: bool, use randomized samples.
        ray_shape: string, which kind of shape to assume for the ray.
        stop_grad: bool, whether or not to backprop through sampling.
        resample_padding: float, added to the weights before normalizing.
    Returns:
        t_samples: torch.Tensor, [batch_size, num_samples+1].
        points: torch.Tensor, [batch_size, num_samples, 3].
    """
    # Do a blurpool.
    if stop_grad:
        with torch.no_grad():
            weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
            weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
            weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

            # Add in a constant (the sampling function will renormalize the PDF).
            weights = weights_blur + resample_padding

            new_t_vals = sorted_piecewise_constant_pdf(
                t_samples,
                weights,
                t_samples.shape[-1],
                randomized,
            )
    else:
        weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
        weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
        weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

        # Add in a constant (the sampling function will renormalize the PDF).
        weights = weights_blur + resample_padding

        new_t_vals = sorted_piecewise_constant_pdf(
            t_samples,
            weights,
            t_samples.shape[-1],
            randomized,
        )
    means, covs = cast_rays(new_t_vals, origins, directions, radii, ray_shape)
    return new_t_vals, (means, covs)


def expected_sin(x, x_var):
    """Estimates mean and variance of sin(z), z ~ N(x, var)."""
    # When the variance is wide, shrink sin towards zero.
    y = torch.exp(-0.5 * x_var) * torch.sin(x)  # [B, N, 2*3*L]
    y_var = 0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y ** 2
    y_var = torch.maximum(torch.zeros_like(y_var), y_var)
    return y, y_var


def integrated_pos_enc_360(means_covs):
    P = torch.tensor([[0.8506508, 0, 0.5257311],
                      [0.809017, 0.5, 0.309017],
                      [0.5257311, 0.8506508, 0],
                      [1, 0, 0],
                      [0.809017, 0.5, -0.309017],
                      [0.8506508, 0, -0.5257311],
                      [0.309017, 0.809017, -0.5],
                      [0, 0.5257311, -0.8506508],
                      [0.5, 0.309017, -0.809017],
                      [0, 1, 0],
                      [-0.5257311, 0.8506508, 0],
                      [-0.309017, 0.809017, -0.5],
                      [0, 0.5257311, 0.8506508],
                      [-0.309017, 0.809017, 0.5],
                      [0.309017, 0.809017, 0.5],
                      [0.5, 0.309017, 0.809017],
                      [0.5, -0.309017, 0.809017],
                      [0, 0, 1],
                      [-0.5, 0.309017, 0.809017],
                      [-0.809017, 0.5, 0.309017],
                      [-0.809017, 0.5, -0.309017]]).T
    means, covs = means_covs
    P = P.to(means.device)
    means, x_cov = parameterization(means, covs)
    y = torch.matmul(means, P)
    y_var = torch.sum((torch.matmul(x_cov, P)) * P, -2)
    return expected_sin(torch.cat([y, y + 0.5 * torch.tensor(np.pi)], dim=-1), torch.cat([y_var] * 2, dim=-1))[0]


def integrated_pos_enc(means_covs, min_deg, max_deg, diagonal=True):
    """Encode `means` with sinusoids scaled by 2^[min_deg:max_deg-1].
    Args:
        means_covs:[B, N, 3] a tuple containing: means, torch.Tensor, variables to be encoded.
        covs, [B, N, 3] torch.Tensor, covariance matrices.
        min_deg: int, the min degree of the encoding.
        max_deg: int, the max degree of the encoding.
        diagonal: bool, if true, expects input covariances to be diagonal (full otherwise).
    Returns:
        encoded: torch.Tensor, encoded variables.
    """
    if diagonal:
        means, covs_diag = means_covs
        scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)], device=means.device)  # [L]
        # # [B, N, 1, 3] * [L, 1] = [B, N, L, 3]->[B, N, 3L]
        # y = torch.unsqueeze(means, dim=-2) * torch.unsqueeze(scales, dim=-1)
        # y = y.view(y.shape[0], y.shape[1], -1)
        # # [B, N, 1, 3] * [L, 1] = [B, N, L, 3]->[B, N, 3L]
        # y_var = torch.unsqueeze(covs_diag, dim=-2) * torch.unsqueeze(scales, dim=-1) ** 2
        # y_var = y_var.view(y_var.shape[0], y_var.shape[1], -1)

        y = rearrange(torch.unsqueeze(means, dim=-2) * torch.unsqueeze(scales, dim=-1),
                      'batch sample scale_dim mean_dim -> batch sample (scale_dim mean_dim)')
        # [B, N, 1, 3] * [L, 1] = [B, N, L, 3]->[B, N, 3L]
        y_var = rearrange(torch.unsqueeze(covs_diag, dim=-2) * torch.unsqueeze(scales, dim=-1) ** 2,
                          'batch sample scale_dim cov_dim -> batch sample (scale_dim cov_dim)')
    else:
        means, x_cov = means_covs
        num_dims = means.shape[-1]
        # [3, L]
        basis = torch.cat([2 ** i * torch.eye(num_dims, device=means.device) for i in range(min_deg, max_deg)], 1)
        y = torch.matmul(means, basis)  # [B, N, 3] * [3, 3L] = [B, N, 3L]
        y_var = torch.sum((torch.matmul(x_cov, basis)) * basis, -2)
    # sin(y + 0.5 * torch.tensor(np.pi)) = cos(y) 中国的学生脑子一定出现那句 “奇变偶不变 符号看象限”
    return expected_sin(torch.cat([y, y + 0.5 * torch.tensor(np.pi)], dim=-1), torch.cat([y_var] * 2, dim=-1))[0]


def pos_enc(x, min_deg, max_deg, append_identity=True):
    """The positional encoding used by the original NeRF paper."""
    scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)], device=x.device)
    # [B, 1, 3] * [L, 1] = [B, L, 3] -> [B, 3L]
    xb = rearrange(torch.unsqueeze(x, dim=-2) * torch.unsqueeze(scales, dim=-1),
                   'batch scale_dim x_dim -> batch (scale_dim x_dim)')
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * torch.tensor(np.pi)], dim=-1))  # [B, 2*3*L]
    if append_identity:
        return torch.cat([x] + [four_feat], dim=-1)  # [B, 2*3*L+3]
    else:
        return four_feat


def volumetric_rendering(rgb, density, t_samples, dirs, white_bkgd, output_t=False):
    """Volumetric Rendering Function.
    Args:
        rgb: torch.Tensor, color, [batch_size, num_samples, 3]
        density: torch.Tensor, density, [batch_size, num_samples, 1].
        t_samples: torch.Tensor, [batch_size, num_samples+1].
        dirs: torch.Tensor, [batch_size, 3].
        white_bkgd: bool.
    Returns:
        comp_rgb: torch.Tensor, [batch_size, 3].
        disp: torch.Tensor, [batch_size].
        acc: torch.Tensor, [batch_size].
        weights: torch.Tensor, [batch_size, num_samples]
    """
    t_mids = 0.5 * (t_samples[..., :-1] + t_samples[..., 1:])
    t_interval = t_samples[..., 1:] - t_samples[..., :-1]  # [B, N]
    # models/mip.py:8 here sample point by multiply the interval with the direction without normalized, so
    # the delta is norm(t1*d-t2*d) = (t1-t2)*norm(d)
    delta = t_interval * torch.linalg.norm(torch.unsqueeze(dirs, dim=-2), dim=-1)
    # Note that we're quietly turning density from [..., 0] to [...].
    density_delta = density[..., 0] * delta

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)
    ], dim=-1))
    weights = alpha * trans  # [B, N]

    comp_rgb = (torch.unsqueeze(weights, dim=-1) * rgb).sum(axis=-2)  # [B, N, 1] * [B, N, 3] -> [B, 3]
    acc = weights.sum(axis=-1)
    distance = (weights * t_mids).sum(axis=-1) / acc
    distance = torch.clamp(torch.nan_to_num(distance), t_samples[:, 0], t_samples[:, -1])
    if white_bkgd:
        comp_rgb = comp_rgb + (1. - torch.unsqueeze(acc, dim=-1))

    if not output_t:
        return comp_rgb, distance, acc, weights
    else:
        return comp_rgb, distance, acc, weights, t_mids


def volumetric_lighting_composing(rgb, density, t_samples, dirs, white_bkgd, output_t=False):
    """Volumetric Lighting Composition Function.
    Args:
        rgb: torch.Tensor, color, [batch_size, num_samples, 3]
        density: torch.Tensor, density, [batch_size, num_samples, 1].
        t_samples: torch.Tensor, [batch_size, num_samples+1].
        dirs: torch.Tensor, [batch_size, 3].
        white_bkgd: bool.
    Returns:
        comp_rgb: torch.Tensor, [batch_size, 3].
        disp: torch.Tensor, [batch_size].
        acc: torch.Tensor, [batch_size].
        weights: torch.Tensor, [batch_size, num_samples]
    """
    t_mids = 0.5 * (t_samples[..., :-1] + t_samples[..., 1:]) # [B, N]
    t_interval = t_samples[..., 1:] - t_samples[..., :-1]  # [B, N]
    # models/mip.py:8 here sample point by multiply the interval with the direction without normalized, so
    # the delta is norm(t1*d-t2*d) = (t1-t2)*norm(d)
    delta = t_interval * torch.linalg.norm(torch.unsqueeze(dirs, dim=-2), dim=-1)
    # Note that we're quietly turning density from [..., 0] to [...].
    density_delta = density[..., 0] * delta

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)
    ], dim=-1))
    weights = alpha * trans  # [B, N]

    attenuation = 1 / (1 + t_mids ** 2) # range [0, 1]
    # [B, N, 1] * [B, N, 1] * [B, N, 3] -> [B, 3]
    comp_rgb = (weights.unsqueeze(-1) * attenuation.unsqueeze(-1) * rgb).sum(axis=-2)
    acc = weights.sum(axis=-1)
    distance = (weights * t_mids).sum(axis=-1) / acc
    distance = torch.clamp(torch.nan_to_num(distance), t_samples[:, 0], t_samples[:, -1])
    if white_bkgd:
        comp_rgb = comp_rgb + (1. - torch.unsqueeze(acc, dim=-1))

    if not output_t:
        return comp_rgb, distance, acc, weights
    else:
        return comp_rgb, distance, acc, weights, t_mids


def rearrange_render_image(rays, chunk_size=4096):
    # change Rays to list: [origins, directions, viewdirs, radii, lossmult, near, far]
    single_image_rays = [getattr(rays, key) for key in Rays_keys]
    val_mask = single_image_rays[-3]

    # flatten each Rays attribute and put on device
    single_image_rays = [rays_attr.reshape(-1, rays_attr.shape[-1]) for rays_attr in single_image_rays]
    # get the amount of full rays of an image
    length = single_image_rays[0].shape[0]
    # divide each Rays attr into N groups according to chunk_size,
    # the length of the last group <= chunk_size
    single_image_rays = [[rays_attr[i:i + chunk_size] for i in range(0, length, chunk_size)] for
                         rays_attr in single_image_rays]
    # get N, the N for each Rays attr is the same
    length = len(single_image_rays[0])
    # generate N Rays instances
    single_image_rays = [Rays(*[rays_attr[i] for rays_attr in single_image_rays]) for i in range(length)]
    return single_image_rays, val_mask


def contract(x):
    # x: [N, 3]
    return (2 - 1 / (torch.norm(x, dim=-1, keepdim=True))) * x / torch.norm(x, dim=-1, keepdim=True)


def parameterization(means, covs):
    '''
    means: [B, N, 3]
    covs: [B, N, 3, 3]
    '''
    contr_mask = (torch.norm(means, dim=-1) > 1).detach()
    contr_means = means[contr_mask]
    contr_covs = covs[contr_mask]
    means_clone = torch.clone(means)
    covs_clone = torch.clone(covs)
    with torch.no_grad():
        jac = vmap(jacrev(contract))(contr_means)
    contr_covs = jac @ contr_covs @ jac.permute([0, 2, 1])
    means_clone[contr_mask] = contract(contr_means)
    covs_clone[contr_mask] = contr_covs
    return means_clone, covs_clone


if __name__ == '__main__':
    # torch.manual_seed(0)
    # batch_size = 4096
    # origins = torch.rand([batch_size, 3])
    # directions = torch.rand(batch_size, 3)
    # radii = torch.rand([batch_size, 1])
    # num_samples = 64
    # near = torch.rand([batch_size, 1])
    # far = torch.rand([batch_size, 1])
    # randomized = True
    # disparity = False
    # ray_shape = 'cone'

    # means = torch.rand(4096, 32, 3, requires_grad=True)
    # convs = torch.rand(4096, 32, 3, 3, requires_grad=True)
    # # print(s.shape)
    # ss = sample_along_rays(origins, directions, radii, num_samples, near, far, randomized, disparity, ray_shape, True)
    # print(ss[0].shape, ss[1][0].shape, ss[1][1].shape)
    # s = integrated_pos_enc_360(ss[1])
    # print(s.shape)
    # ss = mipnerf360_scale(means, 0, 2)
    # print(s)
    # print(jac)
    import matplotlib.pyplot as plt
    def plot_histogram(weights, bins, label):
        bar_center = [(bins[i] + bins[i + 1]) / 2. for i in range(len(bins) - 1)]
        widths = [bins[i + 1] - bins[i] for i in range(len(bins) - 1)]
        plt.bar(bar_center, weights, widths, alpha=0.3, edgecolor='black', label=label)

    ## As batch
    batch_size = 3
    n_samples = 4

    # from MLPprop
    # w_ = torch.rand(batch_size, n_samples)
    # t_ = torch.rand(batch_size, n_samples + 1).sort()[0]
    w_ = torch.tensor([[2, 1, 2.5, 1.1], [2, 1, 2.5, 1.1], [2, 1, 2.5, 1.1]])
    t_ = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])

    # from MLPnerf
    # w = torch.rand(batch_size, n_samples)
    # t = torch.rand(batch_size, n_samples + 1).sort()[0]
    w = torch.tensor([[3, 2.5, 2, 1.8], [3, 2.5, 2, 1.8], [3, 2.5, 2, 1.8]])
    t = torch.tensor([[1, 1.2, 2.3, 4.5, 5], [1, 1.1, 1.5, 1.8, 5], [1, 2, 3, 4.5, 5]])

    # bounds
    ss_tt_ = torch.searchsorted(t, t_)
    ssr_tt_ = torch.searchsorted(t, t_, right=True)

    ssr_tt_below = ssr_tt_ - 1

    # integral or summed-area table of weights here starts with the value 0
    integral_w = torch.cat([torch.zeros([batch_size, 1]), torch.cumsum(w, -1)], -1)

    # indices of integral of weights
    inds = torch.stack([ss_tt_[:, 1:], ssr_tt_below[:, :-1]], -1)

    matched_shape = [inds.shape[0], inds.shape[1], integral_w.shape[-1]]
    integrals = torch.gather(integral_w.unsqueeze(1).expand(matched_shape), 2, inds)

    # calculate the bounds, similarly to summed-area tables
    bounds = integrals[..., 0] - integrals[..., 1]

    # equation 13
    loss = torch.sum(torch.clip(w_ - bounds, 0.0, 1e6)**2 / (w_ + 1e-6))
    print(loss)

    plt.figure()
    i = 0
    plot_histogram(w_[i, :], t_[i, :], 'w_')
    plot_histogram(w[i, :], t[i, :], 'w')
    plot_histogram(bounds[i, :], t_[i, :], 'bounds_')
    plt.legend(loc='upper right')
    plt.show()

    plt.figure()
    i = 1
    plot_histogram(w_[i, :], t_[i, :], 'w_')
    plot_histogram(w[i, :], t[i, :], 'w')
    plot_histogram(bounds[i, :], t_[i, :], 'bounds_')
    plt.legend(loc='upper right')
    plt.show()

    plt.figure()
    i = 2
    plot_histogram(w_[i, :], t_[i, :], 'w_')
    plot_histogram(w[i, :], t[i, :], 'w')
    plot_histogram(bounds[i, :], t_[i, :], 'bounds_')
    plt.legend(loc='upper right')
    plt.show()