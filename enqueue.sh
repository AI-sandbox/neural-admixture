#!/bin/bash
#SBATCH --gres=gpu:1,gpumem:15G
#SBATCH --mem=64G
#SBATCH -o /mnt/gpid08/users/albert.dominguez/logs/chr1/%j.out
#SBATCH -p gpi.compute
#SBATCH --time=12:00:00
source ~/venv/deep_genomics/bin/activate
cd ~/neural-admixture/scripts/src
python3 launch_fit.py --wandb_log 1 --multihead 1 --deep_encoder 1 --activation relu --display_logs 0 --batchnorm 1 --dropout 0.8 --l2_penalty 0.01 --learning_rate 0.01 --batch_size 200 --k 7 --epochs 10 --decoder_init minibatch_kmeans_logit --loss bce --mask_frac 1 --optimizer adam --save_every 50 --save_dir /mnt/gpid08/users/albert.dominguez/weights/chr1 --window_size 1813K --chr 1 --shuffle 1 --min_k 3 --max_k 10 --pooling 1
