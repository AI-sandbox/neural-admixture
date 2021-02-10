#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -o ../logs/ae_%j.out
#SBATCH -p gpi.compute
#SBATCH --time=16:00:00
source ~/venv/deep_genomics/bin/activate
python3 launch_fit.py --lr 0.001 --bs 400 --k 7 --epochs 5000 --lambda_l0 0 --decoder_init kmeans_logit --weight_loss 1 --optimizer adam --save_every 500 --save_dir ../outputs --window_size 50000 --deep_encoder 0
