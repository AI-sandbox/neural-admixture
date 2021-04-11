import allel
from collections import Counter
import gzip
import numpy as np
import os
import gc
import pandas as pd
import logging
import random
import string
import glob
import sys

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# -------------- numpy to vcf utils --------------
def get_name(name_len=8):
    letters = string.ascii_lowercase + string.ascii_uppercase + string.digits
    return ''.join(random.choice(letters) for i in range(name_len)) 

def npy_to_vcf(reference, npy, output_file, filter_samples=None, filter_pos=None, verbose=False):
    """
    - reference: str path to reference file which provides metadata for the results
                 or alternatively, a allel.read_vcf output
    - npy: npy matrix - shape: (num_samples, num_snps)
           make sure npy file has same snp positions
    - output_file: str output vcf path
    
    this is a very light version of npy_to_vcf for LAI applications
    
    Function behavior
    a vcf file called <results_file> with data in npy file and metadata from reference
    - metadata includes all fields except for genotype data
    - npy file must follow convention where maternal and paternal sequences appear one after the other
      for each sample
    NOTE: New to production. Has not been bullet-tested.
    """
    
    if output_file.split(".")[-1] not in ["vcf", "bcf"]:
        output_file += ".vcf"

    log.info('Reading VCF...')
    # read in the input vcf data
    if type(reference) == str:
        data = allel.read_vcf(reference)
    else:
        data = reference.copy()
    
    log.info('VCF read.')
    
    # infer chromosome length and number of samples
    npy = npy.astype(int)
    chmlen, _, _ = data["calldata/GT"].shape
    h, c = npy.shape
    n = h//2
    assert chmlen == c, "reference (" + str(chmlen) + ") and numpy matrix (" + str(c) + ") not compatible"

    if filter_samples is not None:
        N = []
        for el in filter_samples:
            N.append(list(data['samples']).index(el))
        data['samples'] = data['samples'][N]
        
    log.info('Number of samples is {}'.format(len(data["samples"])))
    log.info(f'n is {n}')
    
    # Keep sample names if appropriate
    if "samples" in list(data.keys()) and len(data["samples"]) == n:
        if verbose:
            log.info("Using same sample names")
        data_samples = data["samples"]
    else:
        data_samples = [get_name() for _ in range(n)]
        
    if filter_pos is not None:
        pos_filter_list = np.loadtxt(filter_pos)
        pos_all = data["variants/POS"]
        indx_pos = np.where(np.isin(pos_all, pos_filter_list))[0]
        
        chmlen = len(indx_pos)
        c = len(indx_pos)
        data["variants/CHROM"] = data["variants/CHROM"][indx_pos]
        data["variants/POS"] = data["variants/POS"][indx_pos]
        data["variants/ID"] = data["variants/ID"][indx_pos]
        data["variants/REF"] = data["variants/REF"][indx_pos]
        data["variants/ALT"] = data["variants/ALT"][indx_pos]
        data["variants/QUAL"] = data["variants/QUAL"][indx_pos]


      
    log.info(data["variants/CHROM"])
    if 'chr' in data["variants/CHROM"][0]:
        for j in range(len(data["variants/CHROM"])):
            data["variants/CHROM"][j] = data["variants/CHROM"][j].replace('chr', '')

    # metadata 
    df = pd.DataFrame()
    df["CHROM"]  = data["variants/CHROM"]
    df['POS']    = data["variants/POS"]
    df["ID"]     = data["variants/ID"]
    df["REF"]    = data["variants/REF"]
    df["VAR"]    = data["variants/ALT"][:,0] # ONLY THE FIRST SINCE WE ONLY CARE ABOUT BI-ALLELIC SNPS HERE FOR NOW
    df["QUAL"]   = data["variants/QUAL"]
    df["FILTER"] = ["PASS"]*chmlen
    df["INFO"]   = ["."]*chmlen
    df["FORMAT"] = ["GT"]*chmlen
    del data
    gc.collect()
    # genotype data for each sample
    for i in range(n):
        if i % 50 == 0:
            log.info(f'{i} of {n}')
        # get that particular individual's maternal and paternal snps
        maternal = npy[i*2,:].astype(str) # maternal is the first
        paternal = npy[i*2+1,:].astype(str) # paternal is the second

        # create "maternal|paternal"
        lst = [maternal, ["|"]*chmlen, paternal]
        genotype_person = list(map(''.join, zip(*lst)))
        del paternal, maternal
        df[data_samples[i]] = genotype_person

    if verbose:
        log.info(f'Writing vcf data in {output_file}')

    # write header
    with open(output_file,"w") as f:
        f.write("##fileformat=VCFv4.1\n")
        f.write("##source=pyadmix (XGMix)\n")
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased Genotype">\n')
        f.write("#"+"\t".join(df.columns)+"\n") # mandatory header
    
    # genotype data
    df.to_csv(output_file,sep="\t",index=False,mode="a",header=False)
    return

if __name__ == '__main__':
    data_path = '/mnt/gpid08/users/albert.dominguez/data/chr22'
    ancestries = ['AFR', 'AMR', 'EAS', 'EUR', 'OCE', 'SAS', 'WAS']
    which_set = 'valid'
    window_size = 317408
    log.info('Fetching SNPs arrays...')
    for i, ancestry in enumerate(ancestries):
        if i == 0:
            npy = np.empty((0, window_size), int)
        for i, snps_file in enumerate([f'{data_path}/{which_set}/{ancestry}/gen_0/mat_vcf_2d.npy', f'{data_path}/{which_set}/{ancestry}/gen_2/mat_vcf_2d.npy']):
            aux = np.load(snps_file, mmap_mode='r')[:,:window_size]
            npy = np.vstack((npy, aux))
            del aux
            gc.collect()
        log.info(f'{ancestry} SNPs fetched.')
    log.info('All SNPs fetched.')
    reference = f'{data_path}/ref_final_beagle_phased_1kg_hgdp_sgdp_chr22_hg19.vcf.gz'
    results_file = f'{data_path}/{which_set}/{window_size}.vcf'
    npy_to_vcf(reference, npy, results_file, filter_samples=None, filter_pos=None, verbose=True)
