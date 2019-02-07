
from __future__ import print_function

import os
import h5py
import deepdish
from collections import OrderedDict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
plt.style.use("seaborn")



def check_all_none(l):
    """Check all any of the arrays are none"""
    return not any([np.any(a) for a in l])

def check_any_none(l):
    """Check if any of the arrays are none"""
    return not all([np.any(a) for a in l])

def make_plots(outdir, x_val=None, y_val=None, y_pred=None, y_train=None, y_train_pred=None, history=None, parameters=None, loss=None, val_loss=None, KL=None, val_KL=None, scatter=True, **kwargs):
    """
    Make plots from outputs of neural network training for a block of data

    NOTE: arguments correspond to keys of each block in the results file
    """

    if check_all_none([x_val, y_val, y_pred, y_train, y_train_pred, history, loss, val_loss, KL, val_KL]):
        raise ValueError("Not inputs to plot!")

    if not os.path.isdir(outdir):
        raise ValueError("Output directory doesnr't exist!")

    print("Making plots...")
    print("Plots will be saved to: " + outdir)
    if history is not None:
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        KL = history.history["KL"]
        val_KL = history.history["KL"]
    if not check_any_none([loss, val_loss]):
        print("Making metric plots...")
        # loss
        fig = plt.figure()
        plt.plot(loss, label="loss")
        plt.plot(val_loss, label="val loss")
        plt.xlabel("epoch")
        plt.legend()
        fig.savefig(outdir + "loss.png")
        plt.close(fig)
        # log loss
        fig = plt.figure()
        plt.plot(loss, label="loss")
        plt.plot(val_loss, label="val loss")
        plt.xlabel("epoch")
        plt.yscale("log")
        plt.legend()
        fig.savefig(outdir + "loss_log.png")
        plt.close(fig)
    if not check_any_none([KL, val_KL]):
        # kl divergence
        fig = plt.figure()
        plt.plot(KL, label="training")
        plt.plot(val_KL, label="val")
        plt.legend()
        fig.savefig(outdir + "KL.png")
        plt.close(fig)
    if not check_any_none([y_val, y_pred]):
        print("Making prediction plots...")
        # predictions on validation data
        fig = plt.figure()
        plt.plot(y_val, y_pred, ".")
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()])
        #plt.plot([y_val.min(), y_val.max()], [y_val.mean(), y_val.mean()])
        plt.xlabel("Target")
        plt.ylabel("Predicted")
        fig.savefig(outdir + "preds.png")
        plt.close(fig)
    if not check_any_none([y_train, y_train_pred]):
        print("Making prediction plots for training...")
        # predictions for training data
        fig = plt.figure()
        plt.plot(y_train, y_train_pred, ".")
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()])
        #plt.plot([y_train.min(), y_train.max()], [y_train.mean(), y_train.mean()])
        plt.xlabel("Target")
        plt.ylabel("Predicted")
        fig.savefig(outdir + "training_preds.png")
        plt.close(fig)
    if not check_any_none([y_train, y_val]):
        print("Making distribution plots...")
        # histogram of data
        fig = plt.figure()
        plt.hist(y_train, alpha=0.5, label="train", density=True)
        plt.hist(y_val, alpha=0.5, label="val", density=True)
        plt.legend()
        plt.title("True logL")
        fig.savefig(outdir + "data_logL_dist.png")
        plt.close(fig)
    if not check_any_none([y_val, y_pred]):
        print("Making output plots...")
        # histogram of outputs
        fig = plt.figure()
        plt.hist(y_val, alpha=0.5, density=True, label="true")
        plt.hist(y_pred, alpha=0.5, density=True, label="predicted")
        plt.legend()
        plt.title("Output distribution")
        fig.savefig(outdir + "output_dist.png")
        plt.close(fig)
    if not check_any_none([x_val, y_val, y_pred]) and scatter:
        print("Making scatter plot...")
        # scatter of error on logL
        N_params = np.shape(x_val)[-1]
        error = (y_val - y_pred)
        max_error = np.max(np.abs(error))
        fig, axes = plt.subplots(N_params, N_params, figsize=[18, 15])
        for i in range(N_params):
            for j in range(N_params):
                ax = axes[i, j]
                if j < i:
                    idx = [j, i]
                    sp = x_val[:, idx].T
                    sc = ax.scatter(*sp, c=error, vmin=-max_error, vmax=max_error, marker=".", cmap=plt.cm.RdBu_r)
                    # add labels if possible
                    if parameters is not None:
                        if (i + 1) == N_params:
                            ax.set_xlabel(parameters[j])
                        if j == 0:
                            ax.set_ylabel(parameters[i])
                elif j == i:
                    h = x_val[:, j].T
                    ax.hist(h, density=True, alpha=0.5)
                else:
                    ax.set_axis_off()
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.5)
        cbar.set_label("Error logL", rotation=270)
        fig.savefig(outdir + "scatter.png", dpi=400, bbox_inches="tight")
        plt.close(fig)

def read_results(results_path, fname, blocks="all", concat=True, dd=False):
    """Read results saved in blocks"""
    if dd:
        hf = deepdish.io.load(results_path + fname)
    else:
        hf = h5py.File(results_path + fname, "r")
    # need data in order is was added
    d = OrderedDict()
    if blocks == "all":
        blocks = ["block{}".format(b) for b in range(1, len(hf.keys()) + 1)]
    else:
        blocks = ["block{}".format(b) for b in blocks]
    # blocks must be in order for loss plots
    block_keys = sorted(hf.keys(), key=lambda s: int(s.split("block")[-1]))
    for block_key in block_keys:
        if block_key in blocks:
            for key in hf[block_key].keys():
                if key not in d.keys():
                    d[key] = [hf[block_key][key][:]]
                else:
                    d[key].append(hf[block_key][key][:])
    # concatenate all the blocks together
    if concat:
        for key, value in d.iteritems():
            if key is "parameters":
                d[key] = value[0]
            else:
                d[key] = np.concatenate(value, axis=0)

    return d

def get_weights(results_path, weights_fname="model_weights.h5", blocks="all"):
    """
    Return all the weights files for a run

    Args:
        results_path: Path to the directory with the weights file
        weights_fname: Weights file name
        blocks: Blocks to return weights for. Defaults to all and can be a list of intergers, 'all' or 'last'
    Returns:
        weights_files: List of paths to weights files
    """
    run_blocks_unsrt = filter(os.path.isdir, [results_path + i for i in os.listdir(results_path)])
    run_blocks = sorted(run_blocks_unsrt, key=lambda s: int(s.split("block")[-1]))
    if blocks is "all":
        pass
    elif blocks is "last":
        run_blocks = [run_blocks[-1]]
    else:
        run_blocks = []
        for b in run_blocks:
            n = b.split("block")[-1]
            if n in blocks:
                run_blocks.append(b)
    weights_files = []
    for rb in run_blocks:
        wf = rb + "/" + weights_fname
        if os.path.isfile(wf):
            weights_files.append(wf)
    return weights_files

def make_plots_multiple(results_path, outdir, fname='results.h5', blocks='all', plot_function=make_plots, **kwargs):
    """
    Search path for results files with given name and make combined plots.
    """
    # get dictionary of results
    d = read_results(results_path, fname, blocks, concat=True, **kwargs)
    plot_function(outdir, scatter=True, **d)


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
        fig = corner.corner(posterior_samples, labels=parameter_labels)
        fig.savefig(outdir + 'corner.png')


def compare_runs_2d(results_path, outdir, data_path=None, fname="results.h5", model_name="auto_model.json", parameters=["neurons", "layers"], block=3, metric="MSE", labels=None):
    """Compare the results from a directory full of runs using models that vary over two parameters"""

    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    count = 0
    run_path = results_path + "run{}/".format(count)
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
    print("Comparing {} runs".format(count))
    if metric == "MSE":
        train_metric = "loss"
        val_metric = "val_loss"
    else:
        train_metric = metric
        val_metric = "val_" + metric

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
        MaxSE[i] = np.max((r["y_true"][-1][:] - r["y_pred"][-1][:]) ** 2.)

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

    Z = {"train": Z_train, "val": Z_val}

    for key, z in Z.iteritems():
        fig = plt.figure(figsize=(12, 10))
        sc = plt.scatter(X.flatten(), Y.flatten(), c=np.log(z.flatten()), cmap=plt.cm.viridis_r)
        cbar = plt.colorbar(sc)
        cbar.set_label("log " + metric)
        plt.xlabel(X_label)
        plt.ylabel(Y_label)
        plt.xticks(X[:, 0], rotation="45")
        plt.yticks(Y[0, :], rotation="45")
        fig.savefig(outdir + "scatter_{}_{}.png".format(key, metric))

        fig = plt.figure(figsize=(12, 10))
        cf = plt.contourf(X, Y, np.log(z), cmap=plt.cm.viridis_r)
        cbar = plt.colorbar(cf)
        cbar.set_label("log " + metric)
        plt.xlabel(X_label)
        plt.ylabel(Y_label)
        plt.xticks(X[:, 0], rotation="45")
        plt.yticks(Y[0, :], rotation="45")
        fig.savefig(outdir + "contour_{}_{}.png".format(key, metric))

    fig = plt.figure(figsize=(12, 10))
    sc = plt.scatter(X.flatten(), Y.flatten(), c=np.log(MaxSE), cmap=plt.cm.viridis_r)
    cbar = plt.colorbar(sc)
    cbar.set_label("log MaxSE")
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.xticks(X[:, 0], rotation="45")
    plt.yticks(Y[0, :], rotation="45")
    fig.savefig(outdir + "scatter_val_MaxSE.png".format(key, metric))

def compare_to_posterior(posterior_samples, posterior_values, preds, additional_metrics=None):
    """
    Compare the output of the nn with the true posterior values

    By default evaluates:
    * KL divergence
    * Mean squared error
    * Max. squared error

    Args:
        posteior_samples: Parameter values for posterior samples
        posterior_values: Posterior samples
        additional_metrics: List of additional metrics to use

    Returns:
        metrics: A dictionary of metrics
    """
    metrics = {}
    metrics["KL"] = np.sum(posterior_values * np.log(posterior_values / preds))
    metrics["MeanSE"] = np.mean((posterior_values - preds) ** 2.)
    metrics["MaxSE"] = np.max((posterior_values - preds) ** 2.)

    if additional_metrics is not None:
        for name, f in additional_metrics.iteritems():
            metrics[name] = f(posterior_values, preds)
    return metrics

def compare_run_to_posterior(run_path, sampling_results, outdir=None, fname="results.h5", plots=True, additional_metrics=None):
    """
    Load a weights for model and compare the predicted values to the true posterior

    See compare_to_posterior for metrics that are evaluated by default

    Args:
        run_path: Path to the run
        sampling_results: path to results from bilby sampling
        outdir: output directory, if None defaults to run_path
        fname: name of results file in run directory
        plots: enable or disable plots
        additional_metrics: dict of additional metrics that take y_true and y_pred and return a single value
    Returns:
        data: list of tuples containing the predicted values and metrics dict

    """
    from ..function_approximator import FunctionApproximator
    if outdir is None:
        outdir = run_path
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    # format of posteior results determined by Bilby
    posterior_results = deepdish.io.load(sampling_results)["posterior"]
    p = ["psi", "luminosity_distance", "iota"]
    posterior_samples = posterior_results[p].values
    posterior_values = posterior_results["logL"].values# + posterior_results["logPrior"].values
    FA = FunctionApproximator(attr_dict=run_path + "fa.pkl")
    weights_files = get_weights(run_path, weights_fname="model_weights.h5", blocks="all")
    data = []
    for c, wf in enumerate(weights_files):
        print("Loading weights from: " +  wf)
        FA.load_weights(wf)
        preds = FA.predict(posterior_samples)
        m = compare_to_posterior(posterior_samples, posterior_values, preds, additional_metrics)
        data.append((preds, m))

    if plots:
        print("Making plots")
        n_subplots = int(np.ceil(np.sqrt(len(weights_files))))
        hist_fig = plt.figure(figsize=(12, 10))
        meanSE = np.empty(c + 1)
        for i, d in enumerate(data):
            meanSE[i] = d[1]["MeanSE"]
            ax = hist_fig.add_subplot(n_subplots, n_subplots, i + 1)
            ax.hist(d[0], alpha=0.5, label="Predicted posteior", density=True)
            ax.hist(posterior_values, alpha=0.5, label="True posterior", density=True)
            ax.set_title("Block " + str(i))
            ax.set_xlabel("logL")
            ax.legend(["Predicted", "True"])
        hist_fig.tight_layout()
        hist_fig.savefig(outdir + "hist.png")
        plt.close(hist_fig)

        metrics_fig = plt.figure(figsize=(12, 10))
        plt.plot(meanSE, 'o')
        plt.yscale("log")
        plt.xlabel("Block")
        plt.ylabel("meanSE")
        metrics_fig.savefig(outdir + "meanSE.png")
        plt.close(metrics_fig)

    return data
