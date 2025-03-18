# import wandb
import torch
from pathlib import Path
from torch.nn.functional import normalize

from models.loss import *
from models.mip import rearrange_render_image
from utils.vis import hotmap, save_results
from utils.surface_rendering import hdr_to_ldr
from utils.metrics import *
from systems.base_system import BaseSystem


class PanoNeRFSystem(BaseSystem):
    def training_step(self, batch, batch_index):
        rays, rgbs, *_ = batch
        ldr_rgb_gt = hdr_to_ldr(rgbs[..., :3], dtype='uint8')

        # ensure same device
        if rays.origins.device != self.env_rays.origins.device:
            self.env_rays = self.train_dataset.put_on_device(self.env_rays, rays.origins.device)

        # forward pass
        if self.global_step >= self.hparams['train.surface_start_step'] and self.hparams['train.surface']:
            enable_surf = True
        else:
            enable_surf = False

        use_ort_loss = True if self.hparams['loss.ort_loss'] > 0 else False
        outputs = self.mip_nerf(
            rays=rays,
            env_rays=self.env_rays,
            randomized=self.train_randomized,
            white_bkgd=self.white_bkgd, 
            enable_surf=enable_surf,
            use_ort_loss=use_ort_loss,
        )

        mask = rays.lossmult

        # loss
        (rgb_c, *_), (rgb_f, _, ort_loss, _, alb, _, sf_rgb, _, _) = outputs

        rgb_c, rgb_f = hdr_to_ldr(rgb_c), hdr_to_ldr(rgb_f)
        vol_coarse = (mask * (rgb_c - ldr_rgb_gt) ** 2).sum() / mask.sum()
        vol_fine = (mask * (rgb_f - ldr_rgb_gt) ** 2).sum() / mask.sum()

        if sf_rgb is not None:
            sf_rgb = hdr_to_ldr(sf_rgb)
            vol_surface = (mask * (sf_rgb - ldr_rgb_gt) ** 2).sum() / mask.sum()
            # self.log('train/loss_vol_surface', vol_surface)
        
        loss = self.hparams['loss.coarse_loss_mult'] * vol_coarse + vol_fine

        if self.global_step >= self.hparams['train.surface_start_step'] and self.hparams['train.surface']:
            loss += self.hparams['loss.surface_loss'] * vol_surface

            if self.hparams['loss.chrom_loss'] > 0:
                chrom = normalize(ldr_rgb_gt, dim=-1)
                chrom_alb = normalize(alb, dim=-1)

                chrom_loss = ((chrom - chrom_alb) ** 2).mean()
                loss += self.hparams['loss.chrom_loss'] * chrom_loss
                # self.log('train/loss_chrom', chrom_loss)

        if ort_loss is not None:
            loss += self.hparams['loss.ort_loss'] * ort_loss
            # self.log('train/loss_orientation', ort_loss)

        # self.log('params/lr', self.optimizers().optimizer.param_groups[0]['lr'])
        # self.log('train/loss', loss)
        # self.log('train/loss_vol_coarse', vol_coarse)
        # self.log('train/loss_vol_fine', vol_fine)

        return loss

    def validation_step(self, batch, batch_nb):
        # _, rgbs, dep_gt, nor_gt, alb_gt = batch
        _, gt_rgbs, gt_depth, gt_normal, gt_albedo = batch

        # hdr to ldr
        gt_hdr_image = gt_rgbs[..., :3].permute(0, 3, 1, 2) # (n, c, h, w)
        gt_ldr_image = hdr_to_ldr(gt_hdr_image)
        # normal
        gt_normal = F.normalize(gt_normal, dim=-1).permute(0, 3, 1, 2)
        gt_depth = gt_depth.permute(0, 3, 1, 2)

        _, pred_hdr_image, _, pred_depth, pred_normal, pred_albedo, _, pred_hdr_image_surf, pred_shading = \
            self.render_image(batch)

        # hdr results
        pred_ldr_image = hdr_to_ldr(pred_hdr_image, dtype='uint8')
    
        # geometry results
        near, far = self.hparams['range']
        gt_normal = F.normalize(gt_normal, dim=1)
        pred_normal = F.normalize(pred_normal, dim=1)

        pred_normal = (pred_normal + 1) / 2
        gt_normal = (gt_normal + 1) / 2
        gt_depth_norm = (gt_depth - near) / (far - near)
        pred_depth_norm = (pred_depth - near) / (far - near)
        gt_depth_norm = hotmap(gt_depth_norm)
        pred_depth_norm = hotmap(pred_depth_norm)

        # hdr results from surface rendering
        if pred_hdr_image_surf is not None:
            pred_ldr_image_surf = hdr_to_ldr(pred_hdr_image_surf, dtype='uint8')
        else:
            pred_ldr_image_surf = torch.zeros_like(pred_ldr_image)
            pred_albedo = torch.zeros_like(gt_albedo)

        save_dir = os.path.join(self.hparams['save_dir'], f'val_{self.global_step:06d}')
        save_dir = Path(save_dir)
        # saving hdr resylts
        save_results(gt_hdr_image, save_path=save_dir / 'gt_hdr' / f'{batch_nb:03d}.exr')
        save_results(pred_hdr_image, save_path=save_dir / 'pred_hdr' / f'{batch_nb:03d}.exr')
        save_results(gt_ldr_image, save_path=save_dir / 'gt_ldr' / f'{batch_nb:03d}.png')
        save_results(pred_ldr_image, save_path=save_dir / 'pred_ldr' / f'{batch_nb:03d}.png')

        # saving geometry results
        save_results(gt_normal, save_path=save_dir / 'gt_normal' / f'{batch_nb:03d}.png')
        save_results(pred_normal, save_path=save_dir / 'pred_normal' / f'{batch_nb:03d}.png')
        save_results(gt_depth_norm, save_path=save_dir / 'gt_depth' / f'{batch_nb:03d}.png')
        save_results(pred_depth_norm, save_path=save_dir / 'pred_depth' / f'{batch_nb:03d}.png')
        
        # save hdr and albedo from surface rendering
        if pred_hdr_image_surf is not None:
            save_results(pred_hdr_image_surf, save_path=save_dir / 'pred_hdr_surf' / f'{batch_nb:03d}.exr')
            save_results(pred_ldr_image_surf, save_path=save_dir / 'pred_ldr_surf' / f'{batch_nb:03d}.png')
            save_results(pred_albedo, save_path=save_dir / 'pred_albedo' / f'{batch_nb:03d}.png')

    def render_image(self, batch):
        rays, rgbs = batch[:2]
        if rays.origins.device != self.env_rays.origins.device:
            self.env_rays = self.train_dataset.put_on_device(self.env_rays, rays.origins.device)

        _, height, width, _ = rgbs.shape  # N H W C
        single_image_rays, _ = rearrange_render_image(rays, self.val_chunk_size)
        coarse_rgb, fine_rgb = [], []
        coarse_dep, fine_dep = [], []
        fine_nor, albedo, roughness, surface_rgb, shading = [], [], [], [], []
        with torch.no_grad():
            for batch_rays in single_image_rays:
                (c_rgb, c_dep, *_), (f_rgb, f_dep, _, f_nor, alb, rhn, sf_rgb, _, sd) = \
                    self.mip_nerf(
                    rays=batch_rays,
                    env_rays=self.env_rays,
                    randomized=self.val_randomized,
                    white_bkgd=self.white_bkgd, 
                    enable_surf=True,
                    use_ort_loss=True,
                )

                coarse_rgb.append(c_rgb)
                fine_rgb.append(f_rgb)
                coarse_dep.append(c_dep)
                fine_dep.append(f_dep)
                fine_nor.append(f_nor)

                if alb is not None:
                    albedo.append(alb)
                    surface_rgb.append(sf_rgb)

                if rhn is not None:
                    roughness.append(rhn)

                if sd is not None:
                    shading.append(sd)

        def compose(_x, dim=3):
            return torch.cat(_x, dim=0).view(1, height, width, dim).permute(0, 3, 1, 2)

        coarse_rgb = compose(coarse_rgb)
        fine_rgb = compose(fine_rgb)
        coarse_dep = compose(coarse_dep, dim=1)
        fine_dep = compose(fine_dep, dim=1)
        fine_nor = compose(fine_nor)
        if len(albedo) > 0:
            albedo = compose(albedo)
            surface_rgb = compose(surface_rgb)
        else:
            albedo = None
            surface_rgb = None

        if len(roughness) > 0:
            roughness = compose(roughness, dim=1)
        if len(shading) > 0:
            shading = compose(shading)

        return coarse_rgb, fine_rgb, coarse_dep, fine_dep, \
               fine_nor, albedo, roughness, surface_rgb, shading
    