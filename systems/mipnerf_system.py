# import wandb
from pathlib import Path

from models.loss import *
from models.mip import rearrange_render_image
from utils.vis import hotmap, save_results
from utils.surface_rendering import hdr_to_ldr
from utils.metrics import *
from systems.base_system import BaseSystem


class MipNeRFSystem(BaseSystem):
    def forward(self, batch_rays, randomized, white_bkgd, use_ort_loss=False):
        res = self.mip_nerf(
            rays=batch_rays,
            randomized=randomized,
            white_bkgd=white_bkgd, 
            use_ort_loss=use_ort_loss,
        )
        return res

    def training_step(self, batch, batch_nb):
        rays, rgbs, _, _, _ = batch
        ldr_rgb_gt = hdr_to_ldr(rgbs[..., :3], dtype='uint8')

        use_ort_loss = True if self.hparams['loss.ort_loss'] > 0 else False
        outputs = self.mip_nerf(
            rays=rays,
            randomized=self.train_randomized,
            white_bkgd=self.white_bkgd, 
            use_ort_loss=use_ort_loss,
        )

        mask = rays.lossmult

        (vol_c, *_), (vol_f, _, ort_loss, _) = outputs
        vol_c, vol_f = hdr_to_ldr(vol_c), hdr_to_ldr(vol_f)
        vol_coarse = (mask * (vol_c - ldr_rgb_gt) ** 2).sum() / mask.sum()
        vol_fine = (mask * (vol_f - ldr_rgb_gt) ** 2).sum() / mask.sum()

        loss = self.hparams['loss.coarse_loss_mult'] * vol_coarse + vol_fine
        if use_ort_loss:
            loss += self.hparams['loss.ort_loss'] * ort_loss
        else:
            ort_loss = 0.0

        # self.log('params/lr', self.optimizers().optimizer.param_groups[0]['lr'])
        # self.log('train/loss', loss, prog_bar=True)
        # self.log('train/loss_vol_coarse', vol_coarse)
        # self.log('train/loss_vol_fine', vol_fine)
        # self.log('train/loss_orientation', ort_loss)

        return loss

    def validation_step(self, batch, batch_nb):
        _, gt_rgbs, gt_depth, gt_normal, _ = batch
        # hdr to ldr
        gt_hdr_image = gt_rgbs[..., :3].permute(0, 3, 1, 2) # (n, c, h, w)
        gt_ldr_image = hdr_to_ldr(gt_hdr_image)
        # normal
        gt_normal = F.normalize(gt_normal, dim=-1).permute(0, 3, 1, 2)
        gt_depth = gt_depth.permute(0, 3, 1, 2)

        _, pred_hdr_image, _, pred_depth, _, pred_normal = self.render_image(batch)

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

    def render_image(self, batch):  
        rays, rgbs = batch[:2]

        _, height, width, _ = rgbs.shape  # N H W C
        single_image_rays, _ = rearrange_render_image(rays, self.val_chunk_size)
        vol_coarse, vol_fine, dep_coarse, dep_fine, denor_coarse, denor_fine = [], [], [], [], [], []
        with torch.no_grad():
            for batch_rays in single_image_rays:
                (vol_c, dep_c, _, denor_c), (vol_f, dep_f, _, denor_f) = self.mip_nerf(
                    rays=batch_rays,
                    randomized=self.val_randomized,
                    white_bkgd=self.white_bkgd, 
                    use_ort_loss=True,
                )

                vol_coarse.append(vol_c)
                vol_fine.append(vol_f)
                dep_coarse.append(dep_c)
                dep_fine.append(dep_f)
                denor_coarse.append(denor_c)
                denor_fine.append(denor_f)

        def compose(_x, dim=3):
            return torch.cat(_x, dim=0).view(1, height, width, dim).permute(0, 3, 1, 2)

        vol_coarse = compose(vol_coarse)
        vol_fine = compose(vol_fine)
        dep_coarse = compose(dep_coarse, 1)
        dep_fine = compose(dep_fine, 1)
        denor_coarse = compose(denor_coarse)
        denor_fine = compose(denor_fine)

        return vol_coarse, vol_fine, dep_coarse, dep_fine, denor_coarse, denor_fine