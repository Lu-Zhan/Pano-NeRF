import torch
from collections import namedtuple
from torch.func import jacrev, vmap
from torch.nn.functional import normalize

from models.mip import *
from utils.surface_rendering import *


def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)


class MLP(torch.nn.Module):
    """
    A simple MLP.
    """

    def __init__(self, net_depth: int, net_width: int, net_depth_condition: int, net_width_condition: int,
                 skip_index: int, num_rgb_channels: int, num_density_channels: int, activation: str,
                 xyz_dim: int, view_dim: int):
        """
          net_depth: The depth of the first part of MLP.
          net_width: The width of the first part of MLP.
          net_depth_condition: The depth of the second part of MLP.
          net_width_condition: The width of the second part of MLP.
          activation: The activation function.
          skip_index: Add a skip connection to the output of every N layers.
          num_rgb_channels: The number of RGB channels.
          num_density_channels: The number of density channels.
        """
        super(MLP, self).__init__()
        self.skip_index: int = skip_index  # Add a skip connection to the output of every N layers.
        layers = []
        for i in range(net_depth):
            if i == 0:
                dim_in = xyz_dim
                dim_out = net_width
            elif (i - 1) % skip_index == 0 and i > 1:
                dim_in = net_width + xyz_dim
                dim_out = net_width
            else:
                dim_in = net_width
                dim_out = net_width
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            if activation == 'relu':
                layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
            else:
                raise NotImplementedError
        self.layers = torch.nn.ModuleList(layers)
        del layers
        self.density_layer = torch.nn.Linear(net_width, num_density_channels)
        _xavier_init(self.density_layer)
        self.extra_layer = torch.nn.Linear(net_width, net_width)  # extra_layer is not the same as NeRF
        _xavier_init(self.extra_layer)
        layers = []
        for i in range(net_depth_condition):
            if i == 0:
                dim_in = net_width + view_dim
                dim_out = net_width_condition
            else:
                dim_in = net_width_condition
                dim_out = net_width_condition
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            if activation == 'relu':
                layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
            else:
                raise NotImplementedError
        self.view_layers = torch.nn.Sequential(*layers)
        del layers
        self.color_layer = torch.nn.Linear(net_width_condition, num_rgb_channels)

    def forward(self, x, view_direction=None):
        """Evaluate the MLP.

        Args:
            x: torch.Tensor(float32), [batch, num_samples, feature], points.
            view_direction: torch.Tensor(float32), [batch, feature], if not None, this
            variable will be part of the input to the second part of the MLP
            concatenated with the output vector of the first part of the MLP. If
            None, only the first part of the MLP will be used with input x. In the
            original paper, this variable is the view direction.

        Returns:
            raw_rgb: torch.Tensor(float32), with a shape of
                [batch, num_samples, num_rgb_channels].
            raw_density: torch.Tensor(float32), with a shape of
                [batch, num_samples, num_density_channels].
        """
        num_samples = x.shape[1]
        inputs = x  # [B, N, 2*3*L]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i % self.skip_index == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
        raw_density = self.density_layer(x)
        if view_direction is not None:
            # Output of the first part of MLP.
            bottleneck = self.extra_layer(x)
            # Broadcast condition from [batch, feature] to
            # [batch, num_samples, feature] since all the samples along the same ray
            # have the same viewdir.
            # view_direction: [B, 2*3*L] -> [B, N, 2*3*L]
            view_direction = repeat(view_direction, 'batch feature -> batch sample feature', sample=num_samples)
            x = torch.cat([bottleneck, view_direction], dim=-1)
            # Here use 1 extra layer to align with the original nerf model.
            x = self.view_layers(x)
        raw_rgb = self.color_layer(x)
        return raw_rgb, raw_density


class PanoMipNeRF(torch.nn.Module):
    """Nerf NN Model with both coarse and fine MLPs."""

    def __init__(
        self, num_samples: int = 128,
        num_levels: int = 2,
        resample_padding: float = 0.01,
        stop_resample_grad: bool = True,
        use_viewdirs: bool = True,
        disparity: bool = False,
        ray_shape: str = 'cone',
        min_deg_point: int = 0,
        max_deg_point: int = 16,
        deg_view: int = 4,
        density_activation: str = 'softplus',
        density_noise: float = 0.,
        density_bias: float = -1.,
        rgb_activation: str = 'sigmoid',
        alb_activation: str = 'sigmoid',
        rgb_padding: float = 0.001,
        disable_integration: bool = False,
        append_identity: bool = True,
        mlp_net_depth: int = 8,
        mlp_net_width: int = 256,
        mlp_net_depth_condition: int = 1,
        mlp_net_width_condition: int = 128,
        mlp_skip_index: int = 4,
        mlp_num_rgb_channels: int = 3,
        mlp_num_density_channels: int = 1,
        mlp_net_activation: str = 'relu',
        solid_angle_height: int = 8,
        solid_angle_width: int = 16,
        num_env_samples: int = 10,
        **kwargs
    ):
        super(PanoMipNeRF, self).__init__()
        self.num_env_samples = num_env_samples
        self.solid_angle = solid_angle_refinement(solid_angle_height, solid_angle_width)
        self.num_levels = num_levels  # The number of sampling levels.
        self.num_samples = num_samples  # The number of samples per level.
        self.disparity = disparity  # If True, sample linearly in disparity, not in depth.
        self.ray_shape = ray_shape  # The shape of cast rays ('cone' or 'cylinder').
        self.disable_integration = disable_integration  # If True, use PE instead of IPE.
        self.min_deg_point = min_deg_point  # Min degree of positional encoding for 3D points.
        self.max_deg_point = max_deg_point  # Max degree of positional encoding for 3D points.
        self.use_viewdirs = use_viewdirs  # If True, use view directions as a condition.
        self.deg_view = deg_view  # Degree of positional encoding for viewdirs.
        self.density_noise = density_noise  # Standard deviation of noise added to raw density.
        self.density_bias = density_bias  # The shift added to raw densities pre-activation.
        self.resample_padding = resample_padding  # Dirichlet/alpha "padding" on the histogram.
        self.stop_resample_grad = stop_resample_grad  # If True, don't backprop across levels')
        mlp_xyz_dim = (max_deg_point - min_deg_point) * 3 * 2
        mlp_view_dim = deg_view * 3 * 2
        mlp_view_dim = mlp_view_dim + 3 if append_identity else mlp_view_dim
        self.mlp = MLP(mlp_net_depth, mlp_net_width, mlp_net_depth_condition, mlp_net_width_condition,
                       mlp_skip_index, mlp_num_rgb_channels, mlp_num_density_channels, mlp_net_activation,
                       mlp_xyz_dim, mlp_view_dim)
        if rgb_activation == 'softplus':  # The RGB activation.
            self.rgb_activation = torch.nn.Softplus()
        else:
            raise NotImplementedError
        if alb_activation == 'sigmoid':  # The Albedo activation.
            self.alb_activation = torch.nn.Sigmoid()
        else:
            raise NotImplementedError
        self.rgb_padding = rgb_padding
        if density_activation == 'softplus':  # Density activation.
            self.density_activation = torch.nn.Softplus()
        else:
            raise NotImplementedError

        # self.enable_dist_att = kwargs['dist_att']
        self.detach_dist = False  # kwargs['detach_dist']
        # self.learn_dist_att = kwargs['learn_dist_att']

        # if self.learn_dist_att:
        #     self.dist_att = torch.nn.Parameter(torch.tensor(kwargs['dist_att_param']), requires_grad=True)
        # else:
        #     self.dist_att = kwargs['dist_att_param']

    def forward(
            self, 
            rays: namedtuple, 
            env_rays: namedtuple, 
            randomized: bool,
            white_bkgd: bool,
            enable_surf: bool,
            use_ort_loss: bool,
        ):
        ret = []
        t_samples, weights = None, None
        for i_level in range(self.num_levels):
            if i_level == 0:
                # Stratified sampling along rays
                t_samples, means_covs = sample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    self.num_samples,
                    rays.near,
                    rays.far,
                    randomized,
                    self.disparity,
                    self.ray_shape,
                )
            else:
                t_samples, means_covs = resample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    t_samples,
                    weights,
                    randomized,
                    self.ray_shape,
                    self.stop_resample_grad,
                    resample_padding=self.resample_padding,
                )

            def compute_graph(mean, var, viewdirs):
                if mean.dim() == 1:
                    viewdirs = viewdirs.view(1, -1)
                    mean = mean.view(1, 1, -1)
                    var = var.view(1, 1, -1)

                means_covs = mean, var
                if self.disable_integration:
                    means_covs = (means_covs[0], torch.zeros_like(means_covs[1]))

                samples_enc = integrated_pos_enc(
                    means_covs,
                    self.min_deg_point,
                    self.max_deg_point,
                )  # samples_enc: [B, N, 2*3*L]  L:(max_deg_point - min_deg_point)

                # Point attribute predictions
                if self.use_viewdirs:
                    viewdirs_enc = pos_enc(
                        viewdirs,
                        min_deg=0,
                        max_deg=self.deg_view,
                        append_identity=True,
                    )
                    raw_rgb, raw_density = self.mlp(samples_enc, viewdirs_enc)
                else:
                    raw_rgb, raw_density = self.mlp(samples_enc)

                # Separate density, albedo, roughness
                raw_roughness = raw_density[:, :, -1:]
                raw_albedo = raw_density[..., 1:-1]
                raw_density = raw_density[..., :1]

                # Add noise to regularize the density predictions if needed.
                if randomized and (self.density_noise > 0):
                    raw_density += self.density_noise * torch.randn(raw_density.shape, dtype=raw_density.dtype)

                # Volumetric rendering.
                rgb = self.rgb_activation(raw_rgb)  # [B, N, 3]
                rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
                density = self.density_activation(raw_density + self.density_bias)  # [B, N, 1]
                # albedo range is referred to NeRFactor
                albedo = self.alb_activation(raw_albedo) * 0.77 + 0.03  # [B, N, 3]
                roughness = self.density_activation(raw_roughness - 1)  # [B, N, 1]

                return rgb, density, albedo, roughness

            rgb, density, albedos, roughnesses = compute_graph(means_covs[0], means_covs[1], rays.viewdirs)

            # directly volume rendering for pixels
            comp_rgb, distance, acc, weights = volumetric_rendering(
                rgb,
                density,
                t_samples,
                rays.directions,
                white_bkgd=white_bkgd,
            )

            normal, surface_rgb, albedo, roughness, specular, diffuse, ort_loss, shading = \
                None, None, None, None, None, None, None, None
            if i_level == 1:
                normalized_weights = torch.unsqueeze(weights, dim=-1) / torch.sum(weights, dim=-1).view(-1, 1, 1)

                # Normals: derivative of density when using fine model
                batched_jac = vmap(jacrev(compute_graph, argnums=0))
                mean, var = means_covs[0].view(-1, 3), means_covs[1].view(-1, 3)
                viewdirs = rays.viewdirs.view(-1, 1, 3).repeat(1, means_covs[0].shape[1], 1).view(-1, 3)
                normals = -batched_jac(mean, var, viewdirs)[1]
                normals = normals.view(means_covs[0].shape[0], means_covs[0].shape[1], 3)
                normals = normalize(normals, dim=-1)
                normal = torch.sum(normalized_weights * normals, dim=1) # [B, 3]
                normal = normalize(normal, dim=-1)

                # normal orientation loss
                if use_ort_loss:
                    dot = torch.bmm(normals, rays.directions.view(-1, 3, 1))    # [B, N, 1]
                    ort_loss = torch.sum(normalized_weights * torch.relu(dot) ** 2, dim=1).mean()
                else:
                    ort_loss = None

                if enable_surf:
                    # Albedos: sum of albedos along ray
                    albedo = torch.sum(normalized_weights * albedos, dim=1)  # [B, 3]

                    # sample lighting on the surface, [B, 3] + [B, 3] * [B, 1] = [B, 3]
                    # detach position
                    if self.detach_dist:
                        origins = rays.origins + rays.directions * distance.view(-1, 1).data
                    else:
                        origins = rays.origins + rays.directions * distance.view(-1, 1)

                    lit_t_samples, lit_means_covs, lit_viewdirs = sample_each_points(
                        origins.view(-1, 1, 3),
                        env_rays.directions,
                        self.num_env_samples,
                        env_rays.near,
                        env_rays.far,
                        env_rays.radii,
                        randomized
                    )

                    # input in nerf model, rgb: [B*Dir, Ne, 3]
                    rgb, density, _, _ = compute_graph(lit_means_covs[0], lit_means_covs[1], lit_viewdirs)

                    # volume rendering for lighting
                    # if self.enable_dist_att:
                    #     env_rgb, env_dep, *_ = \
                    #         volumetric_lighting_composing(rgb, density, lit_t_samples, lit_viewdirs, True)
                    # else:
                    env_rgb, *_ = volumetric_rendering(rgb, density, lit_t_samples, lit_viewdirs, False)
                    env_rgb = env_rgb.view(normal.shape[0], -1, 3)  # [B, Dir, 3]

                    lit_dir = lit_viewdirs.view(env_rgb.shape)

                    # surface rendering
                    surface_rgb, diffuse, _, shading = surface_rendering(
                        env_rgb,
                        albedo,
                        normal,
                        None,
                        lit_dir,
                        rays.viewdirs,
                        env_rays.lossmult,
                        output_sd=True,
                    )

            ret.append((comp_rgb, distance, ort_loss, normal, albedo, roughness, surface_rgb, diffuse, shading))

        return ret

    def get_weights(self, rays: namedtuple):
        """The mip-NeRF Model.
        Args:
            rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
        Returns:
            weights: [B, N, 2], the weights of each level.
        """

        randomized, white_bkgd = False, False
        t_samples, weights = None, None
        for i_level in range(self.num_levels):
            # key, rng = random.split(rng)
            if i_level == 0:
                # Stratified sampling along rays
                t_samples, means_covs = sample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    self.num_samples,
                    rays.near,
                    rays.far,
                    randomized,
                    self.disparity,
                    self.ray_shape,
                )
            else:
                t_samples, means_covs = resample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    t_samples,
                    weights,
                    randomized,
                    self.ray_shape,
                    self.stop_resample_grad,
                    resample_padding=self.resample_padding,
                )

            def compute_graph(mean, var, viewdirs):
                if mean.dim() == 1:
                    viewdirs = viewdirs.view(1, -1)
                    mean = mean.view(1, 1, -1)
                    var = var.view(1, 1, -1)

                means_covs = mean, var
                if self.disable_integration:
                    means_covs = (means_covs[0], torch.zeros_like(means_covs[1]))

                samples_enc = integrated_pos_enc(
                    means_covs,
                    self.min_deg_point,
                    self.max_deg_point,
                )  # samples_enc: [B, N, 2*3*L]  L:(max_deg_point - min_deg_point)

                # Point attribute predictions
                if self.use_viewdirs:
                    viewdirs_enc = pos_enc(
                        viewdirs,
                        min_deg=0,
                        max_deg=self.deg_view,
                        append_identity=True,
                    )
                    raw_rgb, raw_density = self.mlp(samples_enc, viewdirs_enc)
                else:
                    raw_rgb, raw_density = self.mlp(samples_enc)

                # Separate density, albedo, roughness
                raw_roughness = raw_density[:, :, -1:]
                raw_albedo = raw_density[..., 1:-1]
                raw_density = raw_density[..., :1]

                # Add noise to regularize the density predictions if needed.
                if randomized and (self.density_noise > 0):
                    raw_density += self.density_noise * torch.randn(raw_density.shape, dtype=raw_density.dtype)

                # Volumetric rendering.
                rgb = self.rgb_activation(raw_rgb)  # [B, N, 3]
                rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
                density = self.density_activation(raw_density + self.density_bias)  # [B, N, 1]
                # albedo range is referred to NeRFactor
                albedo = self.alb_activation(raw_albedo) * 0.77 + 0.03  # [B, N, 3]
                roughness = self.density_activation(raw_roughness - 1)  # [B, N, 1]

                return rgb, density, albedo, roughness

            rgb, density, albedos, roughnesses = compute_graph(means_covs[0], means_covs[1], rays.viewdirs)

            # directly volume rendering for pixels
            comp_rgb, distance, acc, weights = volumetric_rendering(
                rgb,
                density,
                t_samples,
                rays.directions,
                white_bkgd=white_bkgd,
            )

        return weights.view(-1, weights.shape[-1]), distance.view(-1, 1), \
               means_covs[0].view(-1, weights.shape[-1], 3) # (n, s), (n, 1), (n, s, 3)

    def get_normals(self, rays: namedtuple):
        """The mip-NeRF Model.
        Args:
            rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
            env_rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
            randomized: bool, use randomized stratified sampling.
            white_bkgd: bool, if True, use white as the background (black o.w.).
        Returns:
            ret: list, [*(rgb, distance, acc)]
        """
        randomized = False
        white_bkgd = False
        t_samples, weights = None, None
        for i_level in range(self.num_levels):
            # key, rng = random.split(rng)
            if i_level == 0:
                # Stratified sampling along rays
                t_samples, means_covs = sample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    self.num_samples,
                    rays.near,
                    rays.far,
                    randomized,
                    self.disparity,
                    self.ray_shape,
                )
            else:
                t_samples, means_covs = resample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    t_samples,
                    weights,
                    randomized,
                    self.ray_shape,
                    self.stop_resample_grad,
                    resample_padding=self.resample_padding,
                )

            def compute_graph(mean, var, viewdirs):
                if mean.dim() == 1:
                    viewdirs = viewdirs.view(1, -1)
                    mean = mean.view(1, 1, -1)
                    var = var.view(1, 1, -1)

                means_covs = mean, var
                if self.disable_integration:
                    means_covs = (means_covs[0], torch.zeros_like(means_covs[1]))

                samples_enc = integrated_pos_enc(
                    means_covs,
                    self.min_deg_point,
                    self.max_deg_point,
                )  # samples_enc: [B, N, 2*3*L]  L:(max_deg_point - min_deg_point)

                # Point attribute predictions
                if self.use_viewdirs:
                    viewdirs_enc = pos_enc(
                        viewdirs,
                        min_deg=0,
                        max_deg=self.deg_view,
                        append_identity=True,
                    )
                    raw_rgb, raw_density = self.mlp(samples_enc, viewdirs_enc)
                else:
                    raw_rgb, raw_density = self.mlp(samples_enc)

                # Separate density, albedo, roughness
                raw_roughness = raw_density[:, :, -1:]
                raw_albedo = raw_density[..., 1:-1]
                raw_density = raw_density[..., :1]

                # Add noise to regularize the density predictions if needed.
                if randomized and (self.density_noise > 0):
                    raw_density += self.density_noise * torch.randn(raw_density.shape, dtype=raw_density.dtype)

                # Volumetric rendering.
                rgb = self.rgb_activation(raw_rgb)  # [B, N, 3]
                rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
                density = self.density_activation(raw_density + self.density_bias)  # [B, N, 1]
                # albedo range is referred to NeRFactor
                albedo = self.alb_activation(raw_albedo) * 0.77 + 0.03  # [B, N, 3]
                roughness = self.density_activation(raw_roughness - 1)  # [B, N, 1]

                return rgb, density, albedo, roughness

            rgb, density, albedos, roughnesses = compute_graph(means_covs[0], means_covs[1], rays.viewdirs)

            # directly volume rendering for pixels
            comp_rgb, distance, acc, weights = volumetric_rendering(
                rgb,
                density,
                t_samples,
                rays.directions,
                white_bkgd=white_bkgd,
            )

            if i_level == 1:
                normalized_weights = torch.unsqueeze(weights, dim=-1) / torch.sum(weights, dim=-1).view(-1, 1, 1)

                # Normals: derivative of density when using fine model
                batched_jac = vmap(jacrev(compute_graph, argnums=0))
                mean, var = means_covs[0].view(-1, 3), means_covs[1].view(-1, 3)
                viewdirs = rays.viewdirs.view(-1, 1, 3).repeat(1, means_covs[0].shape[1], 1).view(-1, 3)
                normals = -batched_jac(mean, var, viewdirs)[1]
                normals = normals.view(means_covs[0].shape[0], means_covs[0].shape[1], 3)
                normals = normalize(normals, dim=-1)
                normal = torch.sum(normalized_weights * normals, dim=1) # [B, 3]
                normal = normalize(normal, dim=-1)

        return distance.view(-1, 1), normal.view(-1, 3)
