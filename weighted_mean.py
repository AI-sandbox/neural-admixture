import h5py
import numpy as np
import os
import sys
sys.path.append('./scripts/src')
import torch
import logging
from multihead_admixture import AdmixtureMultiHead

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    log.info('Loading model...')
    model_path = '/mnt/gpid08/users/albert.dominguez/weights/2050b02bb7034edbb226106a977f3e3b.pt'
    model = AdmixtureMultiHead([3,4,5,6,7,8,9,10], 317408, batch_norm=True, batch_norm_hidden=False, dropout=0.9)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    log.info('Correct load')
    log.info('Fetching data...')
    tr_file = h5py.File('/mnt/gpid08/users/albert.dominguez/data/chr22/windowed/train317K.h5', 'r')
    val_file = h5py.File('/mnt/gpid08/users/albert.dominguez/data/chr22/windowed/valid317K.h5', 'r')
    trX, trY = tr_file['snps'], tr_file['populations']
    valX, valY = val_file['snps'], val_file['populations']
    log.info('Fetching outputs...')
    tr_outs = torch.load('/mnt/gpid08/users/albert.dominguez/viz/tr_outs_317K_7.pt')
    val_outs = torch.load('/mnt/gpid08/users/albert.dominguez/viz/val_outs_317K_7.pt')
    for i in range(7):
        log.info(f'Computing average for cluster {i}')
        freqs = np.average(trX, weights=tr_outs[:,i], axis=0)
        log.info('MSE Decoder weights vs Frequencies for cluster {}: {}'.format(i, ((torch.tensor(freqs)-[torch.sigmoid(p) for p in model.decoders.decoders[4].parameters()][0][i].detach().cpu())**2).mean()))
    return 0

if __name__ == '__main__':
    sys.exit(main())
