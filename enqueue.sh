#!/bin/bash
#SBATCH --gres=gpu:1,gpumem:10G
#SBATCH --mem=64G
#SBATCH -o /mnt/gpid08/users/albert.dominguez/logs/chr22/%j.out
#SBATCH -p gpi.compute
#SBATCH --time=12:00:00
source ~/venv/deep_genomics/bin/activate
cd ~/neural-admixture/scripts/src
python3 launch_fit.py --wandb_log 1 --multihead 1 --deep_encoder 1 --activation relu --display_logs 0 --batchnorm 1 --dropout 0 --l2_penalty 0.01 --learning_rate 0.0001 --batch_size 200 --k 7 --epochs 10 --decoder_init supervised --loss bce --mask_frac 1 --optimizer adam --save_every 50 --save_dir /mnt/gpid08/users/albert.dominguez/weights/chr22 --chr 22 --shuffle 1 --min_k 7 --max_k 7 --pooling 1 --hidden_size 512 --linear 1 --init_path /mnt/gpid08/users/albert.dominguez/data/chr22/pca_gen2_avg.pkl --freeze_decoder 0 --supervised 1
