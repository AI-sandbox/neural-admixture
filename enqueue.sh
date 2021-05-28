#!/bin/bash
#SBATCH --gres=gpu:1,gpumem:10G
#SBATCH --mem=64G
#SBATCH --nodelist=gpic11
#SBATCH -o /mnt/gpid08/users/albert.dominguez/logs/chr22/%j.out
#SBATCH -p gpi.compute
#SBATCH --time=12:00:00
source ~/venv/deep_genomics/bin/activate
cd ~/neural-admixture/src
python3 launch_fit.py --wandb_log 1 --display_logs 0 --epochs 20 --decoder_init pckmeans --save_dir /mnt/gpid08/users/albert.dominguez/weights/chr22 --chr 22 --hidden_size 512 --linear 1 --init_path /mnt/gpid08/users/albert.dominguez/data/chr22/pca_gen2_avg.pkl --freeze_decoder 0 --supervised 0
