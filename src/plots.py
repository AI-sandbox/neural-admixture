import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import adjusted_mutual_info_score

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

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
    log.info('Classical Mean Squared Second Order Difference (MSSOD): {}'.format(((preds_class@preds_class.T-gt_oh@gt_oh.T)**2).mean()))
    log.info('Neural Mean Squared Second Order Difference (MSSOD): {}'.format(((preds_neur@preds_neur.T-gt_oh@gt_oh.T)**2).mean()))
    log.info('------------------------------------------------------')
    log.info('Classical Adjusted Mutual Information score: {}'.format(adjusted_mutual_info_score(gt, class_assign)))
    log.info('Neural Adjusted Mutual Information score: {}'.format(adjusted_mutual_info_score(gt, neur_assign)))
    log.info('------------------------------------------------------')

def stacked_bar(data, series_labels, category_labels=None, 
                show_values=False, value_format="{}", y_label=None, 
                colors=None, grid=False, reverse=False, legend=False):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will 
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    colors          -- List of color labels
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)
    majority_cluster = np.argmax(data.sum(axis=1))
    data = data[:,data[majority_cluster,:].argsort()[::-1]]
    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        color = colors[i] if colors is not None else None
        axes.append(plt.bar(ind, row_data, bottom=cum_size, 
                            label=series_labels[i], color=color, width=1))
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label)
    plt.yticks([])
    plt.xticks([])
    if legend:
        plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))

    if grid:
        plt.grid()
    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w/2, bar.get_y() + h/2, 
                         value_format.format(h), ha="center", 
                         va="center")
    plt.xlim((0, data.shape[1]))

def plot_pca_multihead(X_pca, y, model, k, Ks, pca, init=None, to_wandb=True):
    ancestries = sorted(np.unique(y).tolist())
    if init is None and model is not None:
        P = [p for p in model.decoders.decoders[k-min(Ks)].parameters()][0].detach().numpy()
    elif init is not None:
        for i in range(k-min(Ks)+1):
            ini = end if i != 0 else 0
            end = ini+Ks[i]
        P = init[ini:end].detach().numpy()
    else:
        raise Exception
    if init is None:
        C = pca.transform(P.T)
    else:
        C = pca.transform(P)
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=0.6, alpha=0.9, cmap="tab10")
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*(scatter.legend_elements()[0], ancestries), title="Classes")

    ax.add_artist(legend1)
    n = [str(i) for i in range(k)]
    for i, txt in enumerate(n):
        ax.annotate(txt, (C[i,0], C[i,1]), fontsize='large')
    plot_key = 'Initialization weights' if init is not None else 'Trained weights'
    plt.title('{}; k = {}'.format(plot_key, k))
    if to_wandb:
        wandb.log({plot_key: wandb.Image(plt)})
    else:
        plt.show()

def generate_plots(model, trX, trY, valX, valY, device,
                   batch_size, k=7, min_k=7, max_k=7,
                   to_wandb=True, epoch=None, pca_obj=None,
                   P_init=None, linear=True, fname='',
                   data_path='', dataset=''):
    model.eval()
    with torch.no_grad():
        tr_outs = []
        for x, _ in model._batch_generator(trX, batch_size, y=None):
            tr_outs.append(model(x.to(device), only_assignments=True)[k-min_k])
        tr_outputs = torch.vstack(tr_outs).detach().cpu().numpy()
        del tr_outs
        val_outs = []
        for x, _ in model._batch_generator(valX, batch_size, y=None):
            val_outs.append(model(x.to(device), only_assignments=True)[k-min_k])
        val_outputs = torch.vstack(val_outs).detach().cpu().numpy()
        del val_outs
    ancestries = ['AFR', 'AMR', 'EAS', 'EUR', 'OCE', 'SAS', 'WAS']
    log.info('Rendering training barplot...')
    plt.figure(figsize=(20,6))
    plt.subplots_adjust(wspace=0, hspace=0)
    for k_idx in range(len(ancestries)):
        if k_idx == 0:
            ax1 = plt.subplot(1,len(ancestries),k_idx+1)
        else:
            plt.subplot(1,len(ancestries),k_idx+1, sharey=ax1)
        labels_plot = [str(i) for i in range(k)]
        stacked_bar(tr_outputs.T[:, np.array(trY) == k_idx], labels_plot, legend=k_idx == len(ancestries)-1)
        plt.title(ancestries[k_idx])
    if to_wandb:
        if epoch is None:
            wandb.log({"Training results": wandb.Image(plt)})
        else:
            wandb.log({"Training results (epoch {})".format(epoch): wandb.Image(plt)})   
    elif fname != '':
        plt.savefig(f'../outputs/figures/{fname}_training_barplot.png')
    log.info('Rendering validation barplot...')
    plt.figure(figsize=(20,6))
    plt.subplots_adjust(wspace=0, hspace=0)
    for k_idx in range(len(ancestries)):
        if k_idx == 0:
            ax1 = plt.subplot(1,len(ancestries),k_idx+1)
        else:
            plt.subplot(1,len(ancestries),k_idx+1, sharey=ax1)
        stacked_bar(val_outputs.T[:, np.array(valY) == k_idx], labels_plot, legend=k_idx == len(ancestries)-1)
        plt.title(ancestries[k_idx])
    if to_wandb:
        if epoch is None:
            wandb.log({"Validation results": wandb.Image(plt)})
        else:
            wandb.log({"Validation results (epoch {})".format(epoch): wandb.Image(plt)})   
    elif fname != '':
        plt.savefig(f'../outputs/figures/{fname}_validation_barplot.png')
    if linear and to_wandb:
        log.info('Rendering PCA plots...')
        try:
            model.to(torch.device('cpu'))
            Ks = np.arange(min_k, max_k+1)
            trX_pca = pca_obj.transform(trX)
            plot_pca_multihead(trX_pca, np.array(trY[:]), None, k, Ks, pca_obj, init=P_init, to_wandb=to_wandb)
            plot_pca_multihead(trX_pca, np.array(trY[:]), model, k, Ks, pca_obj, init=None, to_wandb=to_wandb)
        except Exception as e:
            log.exception(e)
            log.warn('Could not render PCA plots.')
            pass
    elif linear and fname and data_path and dataset:
        name = fname[:6] if 'PRETRAINED' in fname.upper() else fname
        log.info('Computing training metrics...')
        Qs_adm_tr = pd.read_csv(f'{data_path}/{dataset}/{name}_classic_train.Q', sep=' ', names=np.array(range(7))).to_numpy()
        Ps_adm_tr = 1-pd.read_csv(f'{data_path}/{dataset}/{name}_classic_train.P', sep=' ', names=np.array(range(7))).to_numpy().T
        output_metrics(trY[:], Qs_adm_tr, tr_outputs)
        log.info('Computing validation metrics...')
        Qs_adm_val = pd.read_csv(f'{data_path}/{dataset}/{name}_classic_valid.Q', sep=' ', names=np.array(range(7))).to_numpy()
        Ps_adm_val = 1-pd.read_csv(f'{data_path}/{dataset}/{name}_classic_valid.P', sep=' ', names=np.array(range(7))).to_numpy().T
        print_losses(model, valX, Qs_adm_val, Ps_adm_val, device)
        output_metrics(valY[:], Qs_adm_val, val_outputs)
    log.info('Done!')
    return 0
