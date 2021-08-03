import argparse
import logging
import h5py
import numpy as np
import pandas as pd
import sys
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import adjusted_mutual_info_score
from switchers import Switchers
sys.path.append('..')
from model.neural_admixture import NeuralAdmixture

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=False, default='../data', type=str, help='Path to data')
    parser.add_argument('--weights_path', required=False, default='../outputs/weights', type=str, help='Path to trained weights')
    return parser.parse_args()

def get_model_preds(model, data, device):
    model.to(device)
    outs = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for i, (X, _) in enumerate(model._batch_generator(data, 200, shuffle=False)):
            X = X.to(device)
            out = model(X, True)
            outs = torch.cat((outs, out[0].detach().cpu()), axis=0)
    return outs.cpu().numpy()

def print_losses(model, data, Qs_adm, Ps_adm, device, loss_f=torch.nn.BCELoss(reduction='sum')):
    preds_adm = Qs_adm@Ps_adm
    loss_class = 0
    loss_neural = 0 
    model.eval()
    with torch.no_grad():
        for i, (X, _) in enumerate(model._batch_generator(data, 200, shuffle=False)):
            X = X.to(device)
            recs, _ = model(X)
            loss_neural += loss_f(X, recs[0]).item()
            loss_class += loss_f(X, torch.tensor(preds_adm[200*i:200*i+200], dtype=torch.float, device=device)).item()
            del recs
    log.info('Classical loss: {:e}'.format(loss_class))
    log.info('Neural ADMIXTURE loss: {:e}'.format(loss_neural))
    log.info('------------------------------------------------------')

def output_metrics(gt, preds_class, preds_neur):
    assert len(gt) == len(preds_class) == len(preds_neur), 'GT and predictions do not have same number of samples'
    class_assign = np.argmax(preds_class, axis=1)
    neur_assign = np.argmax(preds_neur, axis=1)
    gt_oh = OneHotEncoder().fit_transform(gt.reshape(-1,1))
    log.info('Classical Mean Squared Second Order Difference (Delta): {}'.format(((preds_class@preds_class.T-gt_oh@gt_oh.T)**2).mean()))
    log.info('Neural Mean Squared Second Order Difference (Delta): {}'.format(((preds_neur@preds_neur.T-gt_oh@gt_oh.T)**2).mean()))
    log.info('------------------------------------------------------')
    log.info('Classical Adjusted Mutual Information score (AMI): {}'.format(adjusted_mutual_info_score(gt, class_assign)))
    log.info('Neural Adjusted Mutual Information score(AMI): {}'.format(adjusted_mutual_info_score(gt, neur_assign)))
    log.info('------------------------------------------------------')

def main():
    args = parse_args()
    data_path = args.data_path
    dataset = 'CHM-22'
    switchers = Switchers.get_switchers()
    tr_file, val_file = switchers['data']['CHM-22'](data_path)
    f_tr = h5py.File(tr_file, 'r')
    f_val = h5py.File(val_file, 'r')
    trX, trY, valX, valY = f_tr['snps'], f_tr['populations'], f_val['snps'], f_val['populations']
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    name = 'CHM-22'
    log.info(f'---------------------- {name} ----------------------')
    model_path = f'{args.weights_path}/{name}.pt'
    model = NeuralAdmixture([7], num_features=trX.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=True)
    model.to(device)
    log.info('Computing training metrics...')
    Qs_adm_tr = pd.read_csv(f'{data_path}/{dataset}/{name}_classic_train.Q', sep=' ', names=np.array(range(7))).to_numpy()
    Ps_adm_tr = 1-pd.read_csv(f'{data_path}/{dataset}/{name}_classic_train.P', sep=' ', names=np.array(range(7))).to_numpy().T
    tr_outputs = get_model_preds(model, trX, device)
    output_metrics(trY[:], Qs_adm_tr, tr_outputs)
    log.info('Computing validation metrics...')
    Qs_adm_val = pd.read_csv(f'{data_path}/{dataset}/{name}_classic_valid.Q', sep=' ', names=np.array(range(7))).to_numpy()
    Ps_adm_val = 1-pd.read_csv(f'{data_path}/{dataset}/{name}_classic_valid.P', sep=' ', names=np.array(range(7))).to_numpy().T
    val_outputs = get_model_preds(model, valX, device)
    print_losses(model, valX, Qs_adm_val, Ps_adm_val, device)
    output_metrics(valY[:], Qs_adm_val, val_outputs)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    name = 'CHM-22-PRETRAINED'
    log.info(f'---------------------- {name} ----------------------')
    model_path = f'{args.weights_path}/{name}.pt'
    model = NeuralAdmixture([7], num_features=trX.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=True)
    model.to(device)
    log.info('Computing training metrics...')
    Qs_adm_tr = pd.read_csv(f'{data_path}/{dataset}/CHM-22_classic_train.Q', sep=' ', names=np.array(range(7))).to_numpy()
    Ps_adm_tr = 1-pd.read_csv(f'{data_path}/{dataset}/CHM-22_classic_train.P', sep=' ', names=np.array(range(7))).to_numpy().T
    tr_outputs = get_model_preds(model, trX, device)
    output_metrics(trY[:], Qs_adm_tr, tr_outputs)
    log.info('Computing validation metrics...')
    Qs_adm_val = pd.read_csv(f'{data_path}/{dataset}/CHM-22_classic_valid.Q', sep=' ', names=np.array(range(7))).to_numpy()
    Ps_adm_val = 1-pd.read_csv(f'{data_path}/{dataset}/CHM-22_classic_valid.P', sep=' ', names=np.array(range(7))).to_numpy().T
    print_losses(model, valX, Qs_adm_val, Ps_adm_val, device)
    val_outputs = get_model_preds(model, valX, device)
    output_metrics(valY[:], Qs_adm_val, val_outputs)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    name = 'CHM-22-SUPERVISED'
    log.info(f'---------------------- {name} ----------------------')
    model_path = f'{args.weights_path}/{name}.pt'
    model = NeuralAdmixture([7], num_features=trX.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=True)
    model.to(device)
    log.info('Computing training metrics...')
    Qs_adm_tr = pd.read_csv(f'{data_path}/{dataset}/{name}_classic_train.Q', sep=' ', names=np.array(range(7))).to_numpy()
    Ps_adm_tr = 1-pd.read_csv(f'{data_path}/{dataset}/{name}_classic_train.P', sep=' ', names=np.array(range(7))).to_numpy().T
    tr_outputs = get_model_preds(model, trX, device)
    output_metrics(trY[:], Qs_adm_tr, tr_outputs)
    log.info('Computing validation metrics...')
    Qs_adm_val = pd.read_csv(f'{data_path}/{dataset}/{name}_classic_valid.Q', sep=' ', names=np.array(range(7))).to_numpy()
    Ps_adm_val = 1-pd.read_csv(f'{data_path}/{dataset}/{name}_classic_valid.P', sep=' ', names=np.array(range(7))).to_numpy().T
    print_losses(model, valX, Qs_adm_val, Ps_adm_val, device)
    val_outputs = get_model_preds(model, valX, device)
    output_metrics(valY[:], Qs_adm_val, val_outputs)
    return 0

if __name__ == '__main__':
    sys.exit(main())