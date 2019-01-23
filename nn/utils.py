
from __future__ import print_function

import shutil
import os
import json
import numpy as np
import h5py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')

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

############
# Plotting #
############
def check_all_none(l):
    """Check all any of the arrays are none"""
    return not any([np.any(a) for a in l])

def check_any_none(l):
    """Check if any of the arrays are none"""
    return not all([np.any(a) for a in l])

def make_plots(outdir, x=None, y_true=None, y_pred=None, y_train_true=None, y_train_pred=None, history=None, parameters=None, scatter=True):
    """
    Make plots from outputs of neural network training for a block of data
    """

    if check_all_none([x, y_true, y_pred, y_train_true, y_train_pred, history]):
        raise ValueError('Not inputs to plot!')

    if not os.path.isdir(outdir):
        raise ValueError("Output directory doesn't exist!'")

    print('Making plots...')
    print('Plots will be saved to: ' + outdir)
    if history is not None:
        print('Making metric plots...')
        # loss
        fig = plt.figure()
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.xlabel('epoch')
        plt.legend()
        fig.savefig(outdir + 'loss.png')
        plt.close(fig)
        # log loss
        fig = plt.figure()
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.xlabel('epoch')
        plt.yscale('log')
        plt.legend()
        fig.savefig(outdir + 'loss_log.png')
        plt.close(fig)
        # kl divergence
        fig = plt.figure()
        plt.plot(history.history['KL'], label='training')
        plt.plot(history.history['val_KL'], label='val')
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

def make_plots_multiple(results_path, outdir, fname='results.h5', parameters=None, blocks='all'):
    """
    Search path for file results files with given name and make combined plots.
    """
    hf = h5py.File(results_path + fname, 'r')
    d = {}
    if blocks == 'all':
        blocks = ['block{}'.format(b) for b in range(len(hf.keys()))]
    else:
        blocks = ['block{}'.format(b) for b in blocks]
    # load all the blocks into a dictionary
    for block_key in hf.keys():
        if block_key in blocks:
            for key in hf[block_key].keys():
                if key not in d.keys():
                    d[key] = [hf[block_key][key]]
                else:
                    d[key].append(hf[block_key][key])
    # concatenate all the blocks together
    for key, value in d.iteritems():
        if key is 'parameters':
            d[key] = value[0]
        else:
            d[key] = np.concatenate(value, axis=0)

    make_plots(outdir, **d)
