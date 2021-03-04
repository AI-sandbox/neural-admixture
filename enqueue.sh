#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o ./logs/%j.out
#SBATCH -p gpi.compute
#SBATCH --time=12:00:00
source ~/venv/deep_genomics/bin/activate
cd ~/scripts/src
python3 launch_fit.py --wandb_log 0 --display_logs 0 --batchnorm 1 --dropout 0.25 --l2_penalty 0 --lr 0.001 --bs 200 --k 7 --epochs 500 --lambda_l0 0 --decoder_init minibatch_kmeans_logit --loss mse --mask_frac 0.6 --optimizer adam --save_every 10 --save_dir ../../outputs --window_size 100K --deep_encoder 0
