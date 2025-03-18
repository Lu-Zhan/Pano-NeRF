# Pano-NeRF: Synthesizing High Dynamic Range Novel Views with Geometry from Sparse Low Dynamic Range Panoramic Images

## Installation
### Clone the repo
https://github.com/Lu-Zhan/Pano-NeRF.git; cd Pano-NeRF
### Create a conda environment
conda create --name panonerf python=3.9.12; conda activate panonerf
### Prepare pip
conda install pip; pip install --upgrade pip
### Install PyTorch
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
### Install requirements
pip install -r requirements.txt


## Demo Dataset
We provide one demo data `bathroom_0` for testing, which is preprcessed from [Replica dataset](https://github.com/facebookresearch/Replica-Dataset).
Please download the demo datasets from the [Google Drive](https://drive.google.com/drive/folders/1yuTXKQzG26Vn8m81kkyDz28r-nihCViH?usp=sharing) and unzip `bathroom_0.zip` to `data/demo_data/bathroom_0/`.


## Running
To train a model for the `bathroom_0`, please run the following command:
```bash
python train.py --data_path data/demo_data/bathroom_0 --config configs/panonerf.yaml
```
You can also run the script `run.sh` to train all both `mipnerf` and `panonerf` models for `bathroom_0`:
```bash
. scripts/run.sh
```


## Citation
If you find this project useful, please consider citing our paper:
```@inproceedings{lu2024pano,
  title={Pano-NeRF: synthesizing high dynamic range novel views with geometry from sparse low dynamic range panoramic images},
  author={Lu, Zhan and Zheng, Qian and Shi, Boxin and Jiang, Xudong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={4},
  pages={3927--3935},
  year={2024}
}
```


# Acknowledgements
Thansks to [mipnerf](https://github.com/google/mipnerf),
[mipnerf-pytorch](https://github.com/AlphaPlusTT/mipnerf-pytorch),
[nerfplusplus](https://github.com/Kai-46/nerfplusplus),
[nerf_pl](https://github.com/kwea123/nerf_pl),
[mipnerf_pl](https://github.com/hjxwhy/mipnerf_pl),
[Replica dataset](https://github.com/facebookresearch/Replica-Dataset).