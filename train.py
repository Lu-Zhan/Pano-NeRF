import os
import torch
import argparse
import random
import logging
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
# from pytorch_lightning.loggers import WandbLogger

from systems.panonerf_system import PanoNeRFSystem
from systems.mipnerf_system import MipNeRFSystem
from configs.config import parse_args

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
	SystemList = {
		'panonerf': PanoNeRFSystem, 
		'mipnerf': MipNeRFSystem,
	}

	# opinions
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", help="data path.", type=str, default='/home/luzhan/Projects/panonerf/data/bathroom_0')
	parser.add_argument("--out_dir", help="Output directory.", type=str, default='./exps/')
	parser.add_argument("--gpu", help="device number.", nargs='+', type=int, default=[0])
	parser.add_argument("--range", help="device number.", nargs='+', type=int, default=[0, 10])
	parser.add_argument("--dataset_name", help="Single or multi data.", type=str, default='pano_exr')
	parser.add_argument("--valdataset_name", help="Single or multi data.", type=str, default='pano_exr')
	parser.add_argument("--config", help="Path to config file.", required=False, default='./configs/default.yaml')
	parser.add_argument("--project", help="project name", required=False, default='aaai24_formal')
	parser.add_argument("--meta_file", help="meta file name", required=False, default='transforms_all')
	parser.add_argument("--reform_cam", help="if reform campos", type=int, default=0)
	parser.add_argument("opts", nargs=argparse.REMAINDER,
						help="Modify hparams. Example: train.py resume out_dir TRAIN.BATCH_SIZE 2")
	
	hparams = parse_args(parser)

	# prepare
	setup_seed(hparams['seed'])
	hparams['train.sample_num'] = list(map(lambda x: int(x), hparams['train.sample_num'][1:].split('_')))
	hparams['exp_name'] = f"{hparams['nerf.mlp_name']}_{'_'.join(list(map(lambda x: str(x), hparams['train.sample_num'])))}"

	if hparams['train.surface_start_step'] < 1 and hparams['train.surface_start_step'] > 0:
		hparams['train.surface_start_step'] = hparams['train.surface_start_step'] * hparams['optimizer.max_steps']
	
	hparams['save_dir'] = os.path.join(hparams['out_dir'], hparams['exp_name'])
	os.makedirs(hparams['save_dir'], exist_ok=True)
	# logger = WandbLogger(
	# 	save_dir=hparams['save_dir'],
	# 	name=hparams['exp_name'],
	# 	project=hparams['project'],
	# 	log_model=True,
	# )
	# logger.log_hyperparams(hparams)

	ckpt_cb = ModelCheckpoint(
		filename='step={step}-psnr={val_hdr/psnr_hdr_vol:.2f}',
		save_last=True,
		monitor='val_hdr/psnr_hdr_vol',
		mode='max',
		save_top_k=0,
		auto_insert_metric_name=False,
	)
	pbar = TQDMProgressBar(refresh_rate=1)
	callbacks = [ckpt_cb, pbar]

	# setup trainer
	trainer = Trainer(
		max_steps=hparams['optimizer.max_steps'],
		max_epochs=-1,
		callbacks=callbacks,
		check_val_every_n_epoch=hparams['val.check_every_n_epoch'],
		# logger=logger,
		enable_model_summary=False,
		precision='16-mixed',
		accelerator='gpu',
		devices=hparams['gpu'],
		num_sanity_val_steps=1,
		benchmark=True,
		profiler="simple" if len(hparams['gpu']) == 1 else None,
		strategy="ddp" if len(hparams['gpu']) > 1 else "auto",
	)

	# setup training system
	system = SystemList[hparams['nerf.mlp_name']](hparams)
	
	# start training
	trainer.fit(system, ckpt_path=hparams['checkpoint.resume_path'])
