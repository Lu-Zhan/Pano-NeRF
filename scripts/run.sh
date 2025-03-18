scene=bathroom_0

python train.py \
      --data_path /home/luzhan/Projects/panonerf/data/${scene} \
      --config ./configs/panonerf.yaml

python train.py \
      --data_path /home/luzhan/Projects/panonerf/data/${scene} \
      --config ./configs/mipnerf.yaml