
from __future__ import print_function

import shutil
import os
import json
import numpy as np
import h5py
import deepdish

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
plt.style.use('seaborn')

import corner

from scipy.stats import ks_2samp

###########
# General #
###########

def copytree(src, dst, symlinks=False, ignore=None):
    """Move the contents of a directory to a specified directory"""
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def make_run_dir(outdir):
    """Check run count and make outdir"""
    run = 0
    while os.path.isdir(outdir + 'run{}'.format(run)):
        run += 1

    run_path = outdir + 'run{}/'.format(run)
    if not os.path.exists(run_path):
        os.mkdir(run_path)
    return run_path

###########
# Network #
###########

def get_network_params_from_json(model_path, verbose=1):
    """get the parameters for the nn from the model.json file"""
    if verbose:
        print('Loading model: ' + model_path)
    with open(model_path, "r") as read_file:
        params = json.load(read_file)
    return params

############
# Plotting #
############
def check_all_none(l):
    """Check all any of the arrays are none"""
    return not any([np.any(a) for a in l])

def check_any_none(l):
    """Check if any of the arrays are none"""
    return not all([np.any(a) for a in l])

def make_plots(outdir, x=None, y_true=None, y_pred=None, y_train_true=None, y_train_pred=None, history=None, parameters=None, loss=None, val_loss=None, KL=None, val_KL=None, scatter=True):
    """
    Make plots from outputs of neural network training for a block of data
    """

    if check_all_none([x, y_true, y_pred, y_train_true, y_train_pred, history, loss, val_loss, KL, val_KL]):
        raise ValueError('Not inputs to plot!')

    if not os.path.isdir(outdir):
        raise ValueError("Output directory doesn't exist!'")

    print('Making plots...')
    print('Plots will be saved to: ' + outdir)
    if history is not None:
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        KL = history.history['KL']
        val_KL = history.history['val_KL']
    if not check_any_none([loss, val_loss]):
        print('Making metric plots...')
        # loss
        fig = plt.figure()
        plt.plot(loss, label='loss')
        plt.plot(val_loss, label='val loss')
        plt.xlabel('epoch')
        plt.legend()
        fig.savefig(outdir + 'loss.png')
        plt.close(fig)
        # log loss
        fig = plt.figure()
        plt.plot(loss, label='loss')
        plt.plot(val_loss, label='val loss')
        plt.xlabel('epoch')
        plt.yscale('log')
        plt.legend()
        fig.savefig(outdir + 'loss_log.png')
        plt.close(fig)
    if not check_any_none([KL, val_KL]):
        # kl divergence
        fig = plt.figure()
        plt.plot(KL, label='training')
        plt.plot(val_KL, label='val')
        plt.legend()
        fig.savefig(outdir + 'KL.png')
        plt.close(fig)
    if not check_any_none([y_true, y_pred]):
        print('Making prediction plots...')
        # predictions on validation data
        fig = plt.figure()
        plt.plot(y_true, y_pred, '.')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()])
        plt.plot([y_true.min(), y_true.max()], [y_true.mean(), y_true.mean()])
        plt.xlabel('Target')
        plt.ylabel('Predicted')
        fig.savefig(outdir + 'preds.png')
        plt.close(fig)
    if not check_any_none([y_train_true, y_train_pred]):
        print('Making prediction plots for training...')
        # predictions for training data
        fig = plt.figure()
        plt.plot(y_train_true, y_train_pred, '.')
        plt.plot([y_train_true.min(), y_train_true.max()], [y_train_true.min(), y_train_true.max()])
        plt.plot([y_train_true.min(), y_train_true.max()], [y_train_true.mean(), y_train_true.mean()])
        plt.xlabel('Target')
        plt.ylabel('Predicted')
        fig.savefig(outdir + 'training_preds.png')
        plt.close(fig)
    if not check_any_none([y_train_true, y_true]):
        print('Making distribution plots...')
        # histogram of data
        fig = plt.figure()
        plt.hist(y_train_true, alpha=0.5, label='train')
        plt.hist(y_true, alpha=0.5, label='val')
        plt.legend()
        plt.title('True logL')
        fig.savefig(outdir + 'data_logL_dist.png')
        plt.close(fig)
    if not check_any_none([y_true, y_pred]):
        print('Making output plots...')
        # histogram of outputs
        fig = plt.figure()
        plt.hist(y_true, alpha=0.5, normed=True, label='true')
        plt.hist(y_pred, alpha=0.5, normed=True, label='predicted')
        plt.legend()
        plt.title('Output distribution')
        fig.savefig(outdir + 'output_dist.png')
        plt.close(fig)
    if not check_any_none([x, y_true, y_pred]) and scatter:
        print('Making scatter plot...')
        # scatter of error on logL
        N_params = np.shape(x)[-1]
        error = (y_true - y_pred)
        max_error = np.max(np.abs(error))
        fig, axes = plt.subplots(N_params, N_params, figsize=[18, 15])
        for i in range(N_params):
            for j in range(N_params):
                ax = axes[i, j]
                if j <= i:
                    idx = [j, i]
                    sp = x[:, idx].T
                    sc = ax.scatter(*sp, c=error, vmin=-max_error, vmax=max_error, marker='.', cmap=plt.cm.RdBu_r)
                    # add labels if possible
                    if parameters is not None:
                        if (i + 1) == N_params:
                            ax.set_xlabel(parameters[j])
                        if j == 0:
                            ax.set_ylabel(parameters[i])
                else:
                    ax.set_axis_off()
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.5)
        cbar.set_label('Error logL', rotation=270)
        fig.savefig(outdir + 'scatter.png', dpi=400, bbox_inches='tight')
        plt.close(fig)

def read_results(results_path, fname, blocks='all', concat=True):
    """Read results saved in blocks"""
    hf = h5py.File(results_path + fname, 'r')
    d = {}
    if blocks == 'all':
        blocks = ['block{}'.format(b) for b in range(1, len(hf.keys()) + 1)]
    else:
        blocks = ['block{}'.format(b) for b in blocks]
    # load all the blocks into a dictionary
    for block_key in hf.keys():
        if block_key in blocks:
            for key in hf[block_key].keys():
                if key not in d.keys():
                    d[key] = [hf[block_key][key][:]]
                else:
                    d[key].append(hf[block_key][key][:])
    # concatenate all the blocks together
    if concat:
        for key, value in d.iteritems():
            if key is 'parameters':
                d[key] = value[0]
            else:
                d[key] = np.concatenate(value, axis=0)

    return d

def make_plots_multiple(results_path, outdir, fname='results.h5', blocks='all', plot_function=make_plots):
    """
    Search path for file results files with given name and make combined plots.
    """
    # get dictionary of results
    d = read_results(result_path, fname, blocks, concat=True)
    plot_function(outdir, **d)


def compare_runs(results_path, outdir, data_path=None, fname='results.h5', model_name='auto_model.json', parameter='neurons'):
    """Compare runs in a directory"""
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    count = 0
    run_path = results_path + 'run{}/'.format(count)
    results = []
    params = []
    # load all runs
    while os.path.isdir(run_path):
        d = read_results(run_path, fname, blocks='all', concat=False)
        p = get_network_params_from_json(run_path + model_name)
        results.append(d)
        params.append(p)
        run_path = run_path.replace(str(count), str(count + 1))
        count += 1
    # get data arrays
    # get labels
    labels = np.empty(count)
    for i,p in enumerate(params):
        labels[i] = p[parameter]
    # number of blocks
    N_blocks = len(results[0].values()[0])
    MSE = np.empty((N_blocks, count))
    MSE_val = np.empty((N_blocks, count))
    MaxSE = np.empty((N_blocks, count))
    y_true = np.empty((N_blocks, count), dtype=np.ndarray)
    y_pred = np.empty((N_blocks, count), dtype=np.ndarray)
    for b in range(N_blocks):
        for i,d in enumerate(results):
            yt = d['y_true'][b][:]
            yp = d['y_pred'][b][:]
            y_true[b, i] = yt
            y_pred[b, i] = yp
            MaxSE[b, i] = np.max((yt - yp) ** 2.)
            MSE[b, i] = np.min(d['loss'][b])
            MSE_val[b, i] = np.min(d['val_loss'][b])
    # MSE
    fig, axs = plt.subplots(1, 2, figsize=(15,10))
    axs = axs.ravel()
    for b in range(N_blocks):
        axs[0].plot(labels, MSE[b], 'o', label='block' + str(b + 1))
        axs[1].plot(labels, MSE_val[b], 'o', label='block' + str(b + 1))
    for ax in axs:
        ax.set_yscale('log')
        ax.set_ylabel('MSE')
        ax.set_xlabel(parameter)
        ax.set_xticks(labels)
        ax.set_xticklabels(labels, rotation='45')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axs[0].set_title('Training')
    axs[1].set_title('Validation')
    plt.legend()
    fig.savefig(outdir + 'MSE.png')
    plt.close(fig)
    # MaxSE
    fig = plt.figure(figsize=(10, 8))
    for b in range(N_blocks):
        plt.plot(labels, MaxSE[b], 'o', label='block' +  str(b + 1))
    plt.ylabel('Max Squared Error')
    plt.legend()
    plt.yscale('log')
    plt.xticks(labels, rotation='45')
    fig.savefig(outdir + 'MaxSE.png')
    plt.close(fig)

    if data_path is not None:
        output = deepdish.io.load(data_path + 'bilby_result.h5')
        posterior_samples = output['posterior'].drop(['logL', 'logPrior'], axis=1).values
        logL_samples = output['posterior']['logL']
        parameter_labels = output['parameter_labels'][:]
        print(posterior_samples.shape)
        fig = corner.corner(posterior_samples, labels=parameter_labels)
        fig.savefig(outdir + 'corner.png')

    # KS test
    KS_D = np.empty((N_blocks, count), dtype=np.ndarray)
    KS_p = np.empty((N_blocks, count), dtype=np.ndarray)
    for b in range(N_blocks):
        for r in range(count):
            KS_D[b, r], KS_p[b, r] = ks_2samp(logL_samples, y_pred[b, r])
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    axs = axs.ravel()
    for b in range(N_blocks):
        axs[0].plot(labels, KS_D[b, :], 'o', label='block' + str(b + 1))
        axs[1].plot(labels, KS_p[b, :], 'o', label='block' + str(b + 1))
    for ax in axs:
        ax.set_xlabel(parameter)
        ax.set_xticks(labels)
        ax.set_xticklabels(labels, rotation='45')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axs[1].set_yscale('log')
    plt.legend()
    fig.savefig(outdir + 'KS.png')
    plt.close(fig)



def compare_runs_2d(results_path, outdir, data_path=None, fname='results.h5', model_name='auto_model.json', parameters=['neurons', 'layers'], block=3, metric='MSE', labels=None):

    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    count = 0
    run_path = results_path + 'run{}/'.format(count)
    results = []
    params = []
    # load all runs
    while os.path.isdir(run_path):
        d = read_results(run_path, fname, blocks=[block], concat=False)
        p = get_network_params_from_json(run_path + model_name, verbose=0)
        results.append(d)
        params.append(p)
        run_path = run_path.replace(str(count), str(count + 1))
        count += 1
    print('Comparing {} runs'.format(count))
    if metric == 'MSE':
        train_metric = 'loss'
        val_metric = 'val_loss'
    else:
        train_metric = metric
        val_metric = 'val_' + metric

    X = np.empty(count)
    Y = np.empty(count)
    Z_train = np.empty(count)
    Z_val = np.empty(count)
    MaxSE = np.empty(count)
    for i, p, r in zip(range(count), params, results):
        X[i] = p[parameters[0]]
        Y[i] = p[parameters[1]]
        Z_train[i] = np.nanmin(np.abs(r[train_metric][-1][:]))
        Z_val[i] = np.nanmin(np.abs(r[val_metric][-1][:]))
        MaxSE[i] = np.max((r['y_true'][-1][:] - r['y_pred'][-1][:]) ** 2.)

    # labels
    if labels is not None and len(labels) == 2:
        X_label = labels[0]
        Y_label = labels[1]
    else:
        X_label = parameters[0]
        Y_label = parameters[1]
    NX = np.unique(X).shape[0]
    NY = np.unique(Y).shape[0]
    idx = np.argsort(X)
    X = X[idx].reshape(NX, NY)
    Y = Y[idx].reshape(NX, NY)
    Z_train = Z_train[idx].reshape(NX, NY)
    Z_val = Z_val[idx].reshape(NX, NY)

    Z = {'train': Z_train, 'val': Z_val}

    for key, z in Z.iteritems():
        fig = plt.figure(figsize=(12, 10))
        sc = plt.scatter(X.flatten(), Y.flatten(), c=np.log(z.flatten()), cmap=plt.cm.viridis_r)
        cbar = plt.colorbar(sc)
        cbar.set_label('log ' + metric)
        plt.xlabel(X_label)
        plt.ylabel(Y_label)
        plt.xticks(X[:, 0], rotation='45')
        plt.yticks(Y[0, :], rotation='45')
        fig.savefig(outdir + 'scatter_{}_{}.png'.format(key, metric))

        fig = plt.figure(figsize=(12, 10))
        cf = plt.contourf(X, Y, np.log(z), cmap=plt.cm.viridis_r)
        cbar = plt.colorbar(cf)
        cbar.set_label('log ' + metric)
        plt.xlabel(X_label)
        plt.ylabel(Y_label)
        plt.xticks(X[:, 0], rotation='45')
        plt.yticks(Y[0, :], rotation='45')
        fig.savefig(outdir + 'contour_{}_{}.png'.format(key, metric))

    fig = plt.figure(figsize=(12, 10))
    sc = plt.scatter(X.flatten(), Y.flatten(), c=np.log(MaxSE), cmap=plt.cm.viridis_r)
    cbar = plt.colorbar(sc)
    cbar.set_label('log MaxSE')
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.xticks(X[:, 0], rotation='45')
    plt.yticks(Y[0, :], rotation='45')
    fig.savefig(outdir + 'scatter_val_MaxSE.png'.format(key, metric))





