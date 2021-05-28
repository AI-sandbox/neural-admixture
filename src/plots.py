import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

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
                   P_init=None, linear=True):
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
    else:
        plt.show()
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
    else:
        plt.show()
    if linear:
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
    log.info('Done!')
    return 0
