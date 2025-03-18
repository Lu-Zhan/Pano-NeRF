import torch

from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from datasets.pano_datasets import PanoDataset
from utils.lr_schedule import MipLRDecay


class BaseSystem(LightningModule):
    def __init__(self, hparams):
        super(BaseSystem, self).__init__()
        self.save_hyperparameters(hparams)
        self.train_randomized = hparams['train.randomized']
        self.val_randomized = hparams['val.randomized']
        self.white_bkgd = hparams['train.white_bkgd']
        self.val_chunk_size = hparams['val.chunk_size']
        self.batch_size = self.hparams['train.batch_size']

        if self.hparams['nerf.mlp_name'] == 'mipnerf':
            from models.mip_nerf import MipNeRF as NeRFModel
            num_density_channels = 1
        elif self.hparams['nerf.mlp_name'] == 'panonerf':
            from models.pano_mip_nerf import PanoMipNeRF as NeRFModel
            num_density_channels = 5

        self.mip_nerf = NeRFModel(
            num_samples=hparams['nerf.num_samples'],
            num_levels=hparams['nerf.num_levels'],
            resample_padding=hparams['nerf.resample_padding'],
            stop_resample_grad=hparams['nerf.stop_resample_grad'],
            use_viewdirs=hparams['nerf.use_viewdirs'],
            disparity=hparams['nerf.disparity'],
            ray_shape=hparams['nerf.ray_shape'],
            min_deg_point=hparams['nerf.min_deg_point'],
            max_deg_point=hparams['nerf.max_deg_point'],
            deg_view=hparams['nerf.deg_view'],
            density_activation=hparams['nerf.density_activation'],
            density_noise=hparams['nerf.density_noise'],
            density_bias=hparams['nerf.density_bias'],
            rgb_activation=hparams['nerf.rgb_activation'],
            alb_activation=hparams['nerf.alb_activation'],
            rgb_padding=hparams['nerf.rgb_padding'],
            disable_integration=hparams['nerf.disable_integration'],
            append_identity=hparams['nerf.append_identity'],
            mlp_net_depth=hparams['nerf.mlp.net_depth'],
            mlp_net_width=hparams['nerf.mlp.net_width'],
            mlp_net_depth_condition=hparams['nerf.mlp.net_depth_condition'],
            mlp_net_width_condition=hparams['nerf.mlp.net_width_condition'],
            mlp_skip_index=hparams['nerf.mlp.skip_index'],
            mlp_num_rgb_channels=hparams['nerf.mlp.num_rgb_channels'],
            mlp_num_density_channels=num_density_channels,
            mlp_net_activation=hparams['nerf.mlp.net_activation'],
            mlp_name=hparams['nerf.mlp_name'],
            num_env_samples=hparams['nerf.num_env_samples'],
        )

    def setup(self, stage):
        self.train_dataset = PanoDataset(data_dir=self.hparams['data_path'],
                                        split='train',
                                        white_bkgd=self.hparams['train.white_bkgd'],
                                        batch_type=self.hparams['train.batch_type'],
                                        factor=self.hparams['train.factor'],
                                        num=self.hparams['train.sample_num'],
                                        num_start=self.hparams['train.sample_start'],
                                        range=self.hparams['range'],
                                        num_per_epoch=self.hparams['train.batch_size'],
                                        )
        self.val_dataset = PanoDataset(data_dir=self.hparams['data_path'],
                                    split='val',
                                    white_bkgd=self.hparams['val.white_bkgd'],
                                    batch_type=self.hparams['val.batch_type'],
                                    factor=self.hparams['train.factor'],
                                    num=self.hparams['train.sample_num'],
                                    range=self.hparams['range'],
                                    )

        self.env_rays = self.train_dataset.generate_lit_rays(
            num=self.hparams['nerf.num_ray_samples'],
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.mip_nerf.mlp.parameters()),
                                     lr=self.hparams['optimizer.lr_init'])
        scheduler = MipLRDecay(optimizer, self.hparams['optimizer.lr_init'], self.hparams['optimizer.lr_final'],
                               self.hparams['optimizer.max_steps'], self.hparams['optimizer.lr_delay_steps'],
                               self.hparams['optimizer.lr_delay_mult'])
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.hparams['train.num_work'],
            batch_size=self.hparams['train.batch_size'],
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.hparams['val.num_work'],
            batch_size=1,  # validate one image (H*W rays) at a time
            pin_memory=True,
            persistent_workers=True
        )

    def render_normal(self, x, idx):
        x = x.permute((0, 2, 3, 1)) @ torch.tensor(self.val_dataset.obtain_w2c(idx), device=x.device).t()    # (b, h, w, 3) @ (3, 3) = (b, h, w, 3)
        return x.permute((0, 3, 1, 2))

    def clamp_depth(self, depth):
        near, far = self.hparams['range']
        # return (torch.clamp(depth, near, far) - near) / (far - near)
        return torch.clamp(depth, near, far)