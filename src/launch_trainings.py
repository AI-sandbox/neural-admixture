import argparse
import gc
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=False, default='../data', type=str, help='Path where data is stored')
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = args.data_path
    log.info('Training...')
    sps = os.system(f'python3 train.py --display_logs 1 --epochs 10 --decoder_init pckmeans --name CHM-22 --save_dir ../outputs/weights --data_path {data_path} --dataset CHM-22 --linear 1 --init_path ../outputs/initializations/CHM-22_PCA.pkl')
    if sps != 0:
        return sps
    log.info('1/3 models trained.')
    del sps
    gc.collect()
    sps = os.system(f'python3 train.py --display_logs 1 --epochs 10 --decoder_init admixture --name CHM-22-PRETRAINED --save_dir ../outputs/weights --data_path {data_path} --dataset CHM-22 --linear 1 --init_path {data_path}/CHM-22/CHM-22_classic_train.P --freeze_decoder')
    if sps != 0:
        return sps
    log.info('2/3 models trained.')
    del sps
    gc.collect()
    sps = os.system(f'python3 train.py --display_logs 1 --epochs 6 --decoder_init supervised --name CHM-22-SUPERVISED --save_dir ../outputs/weights --data_path {data_path} --dataset CHM-22 --linear 1 --freeze_decoder --supervised')
    if sps != 0:
        return sps
    log.info('3/3 models trained.')
    del sps
    return 0

if __name__ == '__main__':
    Path("../outputs/weights").mkdir(parents=True, exist_ok=True)
    Path("../outputs/initializations").mkdir(parents=True, exist_ok=True)
    Path("../outputs/figures").mkdir(parents=True, exist_ok=True)
    sys.exit(main())