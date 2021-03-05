#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o ./logs/%j.out
#SBATCH -p gpi.compute
#SBATCH --time=24:00:00
source ~/venv/deep_genomics/bin/activate
cd ~/neural-admixture/scripts/src
python3 launch_fit.py --wandb_log 1 --deep_encoder 1 --activation relu --display_logs 0 --batchnorm 1 --dropout 0 --l2_penalty 0 --learning_rate 0.001 --batch_size 200 --k 7 --epochs 500 --decoder_init minibatch_kmeans_logit --loss bce --mask_frac 0.6 --optimizer adam --save_every 25 --save_dir ../../outputs --window_size 100K
