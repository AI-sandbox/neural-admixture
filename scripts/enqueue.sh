#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -o ../logs/ae_deep_gen.out
#SBATCH -p gpi.compute
#SBATCH --time=12:00:00
source ~/venv/deep_genomics/bin/activate
python3 launch_fit.py --lr 0.1 --bs 400 --k 7 --epochs 500 --lambda_l0 0 --decoder_init mean_random --weight_loss 1 --optimizer adam
