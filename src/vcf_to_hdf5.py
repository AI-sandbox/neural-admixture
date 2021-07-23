import allel
import argparse
import h5py
import logging
import numpy as np
import sys

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vcf', required=True, type=str, help='Path of input VCF file.')
    parser.add_argument('--out', required=False, type=str, default='', help='Desired output path for HDF5 file.')
    return parser.parse_args()

def convert_to_hdf5(args):
    vcf_path = args.vcf
    out_path = args.out if args.out != '' else vcf_path.split('.vcf')[0]
    log.info('Reading VCF file...')
    vcf_f = allel.read_vcf(vcf_path)
    log.info('Processing...')
    np_ar = np.sum(vcf_f['calldata/GT'], axis=2).T/2
    log.info('Storing HDF5 file...')
    h5f = h5py.File(out_path, 'w')
    h5f.create_dataset('snps', data=np_ar)
    h5f.close()
    log.info('HDF5 file created.')
    return np_ar
    
def main(args):
    return convert_to_hdf5(args)

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
