
from __future__ import print_function

import os
import six
import h5py
import deepdish
from collections import OrderedDict

import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
plt.style.use("seaborn")
sns.set()
sns.set_style("ticks")

from gwfa.utils import metrics

def check_all_none(l):
    """Check all any of the arrays are none"""
    return not any([np.any(a) for a in l])

def check_any_none(l):
    """Check if any of the arrays are none"""
    return not all([np.any(a) for a in l])

def blank_legend_entry():
    """Return a patch that is not visible in matplotlib legend"""
    return mpl.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor="none",
                                             visible=False)

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
        # scatter of error on logL
        corner_scatter(x_val, y_val, y_pred, outdir=outdir, parameters=parameters)

def corner_scatter(x, y_true, y_pred, outdir='', parameters=None, fname="scatter.png", mkdir=False, save=True, labels_dict=None):
    """
    Make a set of scatter plots for each pair of parameters
    """
    if not os.path.isdir(outdir):
        if mkdir and save:
            os.mkdir(outdir)
        else:
            raise ValueError("Output directory does not exist")
    print("Making scatter plot...")
    # scatter of error on logL
    N_params = np.shape(x)[-1]
    error = (y_true - y_pred)
    max_error = np.max(np.abs(error))
    fig, axes = plt.subplots(N_params-1, N_params-1, figsize=[15, 18])
    for i in range(N_params-1):
        for j in range(N_params-1):
            ax = axes[i, j]
            if j <= i:
                idx = [j, i+1]
                sp = x[:, idx].T
                sc = ax.scatter(*sp, c=error, vmin=-max_error, vmax=max_error, marker=".", cmap=plt.cm.RdYlBu_r)
                ax.xaxis.set_ticks_position("both")
                ax.yaxis.set_ticks_position("both")
            #elif j == i:
                #h = x[:, j].T
                #ax.hist(h, density=True, alpha=0.5, color="firebrick",histtype="stepfilled")
                #ax.xaxis.set_ticks_position("both")
                #ax.yaxis.set_ticks_position("both")
                #ax.set_yticklabels([])

            else:
                ax.set_axis_off()

            if  (i + 2) < N_params:
                ax.get_shared_x_axes().join(ax, axes[i, 0])
                ax.set_xticklabels([])
            else:
                xmin, xmax = ax.get_xlim()
                #ax.set_xticks([xmin, xmax])
                #ax.set_xticklabels(["{:.2}".format(xmin), "{:.2}".format(xmax)], rotation=45)
                if parameters is not None:
                    if labels_dict is not None:
                        ax.set_xlabel(labels_dict[parameters[j]])
                    else:
                        ax.set_xlabel(parameters[j])
            if j > 0:
                ax.get_shared_y_axes().join(ax, axes[0, j])
                ax.set_yticklabels([])
            else:
                ymin, ymax = ax.get_ylim()
                #ax.set_yticks([ymin, ymax])
                #ax.set_yticklabels(["{:.2}".format(ymin), "{:.2}".format(ymax)], rotation=45)
                if parameters is not None:
                    if labels_dict is not None:
                        ax.set_ylabel(labels_dict[parameters[i+1]])
                    else:
                        ax.set_ylabel(parameters[i])
    fig.tight_layout()
    #fig.subplots_adjust(wspace=0.1, hspace=0.1)
    cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat], location="bottom", shrink=0.5, pad=0.09)
    cbar = plt.colorbar(sc, cax=cax, **kw)
    cbar.ax.set_title("Prediction error", )
    if save:
        fig.savefig(outdir + fname, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return fig

def grid_scatter(x, y_true, y_pred, outdir='', parameters=None, fname="scatter.png", mkdir=False, labels_dict=None):
    """
    Make a set of scatter plots for each pair of parameters

    """
    if not os.path.isdir(outdir):
        if mkdir:
            os.mkdir(outdir)
        else:
            print(outdir)
            raise ValueError("Output directory does not exist")
    inputs = [x, y_true, y_pred]
    input_types = []
    for i in inputs:
        if type(i) == list:
            if len(i) == 1:
                i = i[0]
                input_types.append("array")
            elif len(i) > 2:
                raise ValueError("Too arrays in one of the inputs")
            else:
                input_types.append("list")
        elif isinstance(i, np.ndarray):
            input_types.append("array")
    # make sure inputs are all the same type
    if not len(set(input_types)) == 1:
        raise ValueError("Inputs are a mix of arrays and lists")
    else:
        if input_types[0] == "array":
            fig = corner_scatter(*inputs, outdir=outdir, parameters=parameters, fname=fname, mkdir=mkdir, save=False, labels_dict=get_labels_dict())

            fig.savefig(outdir + fname, dpi=400, bbox_inches="tight")
            plt.close(fig)
        else:
            N_params = np.shape(x[0])[-1]
            error = [(t - p) for t, p in zip(y_true, y_pred)]
            max_error = [np.max(np.abs(e), axis=0) for e in error]
            legend_handles = []
            fig, axes = plt.subplots(N_params, N_params, figsize=[18, 15])
            for i in range(N_params):
                for j in range(N_params):
                    ax = axes[i, j]
                    if j < i:# or j > i:
                        idx = [j, i]
                        if j < i:
                            sp = x[0][:, idx].T
                            sc = ax.scatter(*sp, c=error[0], vmin=-max_error[0], vmax=max_error[0], marker=".", cmap=plt.cm.RdBu_r)
                        else:
                            pass
                            for d, c in zip(x, ["blue", "red"]):
                                points = d[:, idx].T
                                #sc = ax.plot(*points, alpha=0.2, marker='.', linestyle='', markersize=4.0)
                                counts, xbins, ybins=np.histogram2d(*points)
                                ct = ax.contour(counts.T, extent=[xbins.min(),xbins.max(), ybins.min(),ybins.max()], alpha=0.5, colors=c)
                        ax.xaxis.set_ticks_position("both")
                        ax.yaxis.set_ticks_position("both")
                    elif j == i:
                        handles = []
                        for e in x:
                            h = e[:, j].T
                            hist = ax.hist(h, density=True, alpha=0.5, histtype="stepfilled")
                            ax.xaxis.set_ticks_position("both")
                            ax.yaxis.set_ticks_position("both")
                            ax.set_yticklabels([])
                        legend_handles.append(handles)
                    else:
                        ax.set_axis_off()
                    if not j == i:
                        if  (i + 1) < N_params:
                            ax.get_shared_x_axes().join(ax, axes[i, 0])
                            if i == 0:
                                ax.xaxis.tick_top()
                                ax.xaxis.set_ticks_position("both")
                            else:
                                ax.set_xticklabels([])
                        if j > 0:
                            ax.get_shared_y_axes().join(ax, axes[0, j])
                            if (j + 1) == N_params and (i + 1) != (N_params):
                                ax.yaxis.tick_right()
                                ax.yaxis.set_ticks_position("both")
                            else:
                                ax.set_yticklabels([])
                    if parameters is not None:
                        if labels_dict is not None:
                            if (i + 1) == N_params:
                                ax.set_xlabel(labels_dict[parameters[j]])
                            if j == 0 and not i == 0:
                                ax.set_ylabel(labels_dict[parameters[i]])
                        else:
                            if (i + 1) == N_params:
                                ax.set_xlabel(parameters[j])
                            elif i == 0:
                                #ax.xaxis.set_label_position("top")
                                #ax.set_xlabel(parameters[j])
                                pass
                            if j == 0 and i != 0:
                                ax.set_ylabel(parameters[i])
                            elif (j + 1) == N_params and (i + 1) != N_params:
                                ax.yaxis.set_label_position("right")
                                ax.set_ylabel(parameters[i])
            plt.tight_layout()
            #fig.subplots_adjust(wspace=0.1, hspace=0.1)
            cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat], location="left", shrink=0.5)
            cbar = plt.colorbar(sc, cax=cax, **kw)
            cbar.set_label("Prediction error")
            #lgd = fig.legend(legend_handles[0], ["Posterior samples", "Training samples"])
            fig.savefig(outdir + fname, dpi=400, bbox_inches="tight")
            plt.close(fig)

def read_results(results_path, fname, blocks="all", concat=True, dd=True):
    """
    Read results saved in blocks

    NOTE: Old runs may be saved with h5py rather than deepdish. For these runs use dd=False.
    """
    if dd:
        hf = deepdish.io.load(results_path + fname)
    else:
        hf = h5py.File(results_path + fname, "r")
    # need data in order is was added
    d = OrderedDict()
    if blocks == "all":
        blocks = ["block{}".format(b) for b in range(0, len(hf.keys()))]
    elif blocks == "last":
        blocks = ["block" +  str(len(hf.keys()) - 1)]
    else:
        blocks = ["block{}".format(b) for b in blocks]
    # blocks must be in order for loss plots
    block_keys = sorted(hf.keys(), key=lambda s: int(s.split("block")[-1]))
    block_keys_used = []
    for block_key in block_keys:
        if block_key in blocks:
            for key in hf[block_key].keys():
                if key not in d.keys():
                    d[key] = [hf[block_key][key][:]]
                else:
                    d[key].append(hf[block_key][key][:])
            block_keys_used.append(block_key)
    # concatenate all the blocks together
    if concat:
        for key, value in six.iteritems(d):
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
    run_blocks_unsrt = filter(lambda s: "block" in s, run_blocks_unsrt)
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

def make_plots_multiple(results_path, outdir, fname='results.h5', blocks='all', plot_function=make_plots, scatter=True, **kwargs):
    """
    Search path for results files with given name and make combined plots.
    """
    # get dictionary of results
    d = read_results(results_path, fname, blocks, concat=True, **kwargs)
    plot_function(outdir, scatter=scatter, **d)


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


def compare_runs_2d(results_path, outdir, run_start=0, data_path=None, fname="results.h5", model_name="auto_model.json", parameters=["neurons", "layers"], block="last", metric="MSE", labels=None):
    """Compare the results from a directory full of runs using models that vary over two parameters"""

    from gwfa.utils import network
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    count = run_start
    run_path = results_path + "run{}/".format(count)
    results = []
    params = []

    # load all runs
    while os.path.isdir(run_path):
        d = read_results(run_path, fname, blocks=block, concat=False)
        p = network.get_parameters_from_json(run_path + model_name, verbose=0)
        results.append(d)
        params.append(p)
        run_path = run_path.replace(str(count), str(count + 1))
        count += 1
    count = count - run_start
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
        MaxSE[i] = np.max((r["y_val"][-1][:] - r["y_pred"][-1][:]) ** 2.)

    # labels
    if labels is not None and len(labels) == 2:
        X_label = labels[0]
        Y_label = labels[1]
    else:
        X_label = parameters[0]
        Y_label = parameters[1]
    NX = np.unique(X).shape[0]
    NY = np.unique(Y).shape[0]
    if len(X) / NX == NY:
        print(len(X), len(Y), NX, NY)
    idx = np.argsort(X)
    X = X[idx].reshape(NX, NY)
    Y = Y[idx].reshape(NX, NY)
    Z_train = Z_train[idx].reshape(NX, NY)
    Z_val = Z_val[idx].reshape(NX, NY)

    Z = {"train": Z_train, "val": Z_val}

    for key, z in six.iteritems(Z):
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
    metrics_dict = OrderedDict()
    P = np.exp(posterior_values)
    Q = np.exp(np.float64(preds))
    metrics_dict["KL"] = metrics.kullback_leibler_divergence(P, Q)
    metrics_dict["JS"] = metrics.jenson_shannon_divergence(P, Q)
    metrics_dict["MeanSE"] = metrics.mean_squared_error(posterior_values, preds)
    metrics_dict["MaxSE"] = metrics.max_squared_error(posterior_values, preds)

    if additional_metrics is not None:
        for name, f in additional_metrics.iteritems():
            metrics_dict[name] = f(posterior_values, preds)
    return metrics_dict

def get_labels_dict():
    """Return a dictionary of labels to use for plots"""
    labels_dict = {"mass_1": r"$m_1$",
            "mass_2": r"$m_2$",
            "psi": r"$\psi$",
            "iota": r"$\iota$",
            "luminosity_distance": r"$d_{\text{L}}$",
            "ra": r"$\alpha$",
            "dec": r"$\delta$",
            "geocent_time": r"$t_{\text{c}}$",
            "tilt_1": r"$t_{1}$",
            "tilt_2": r"$t_{2}$",
            "a_1" : r"$a_{1}$",
            "a_2" : r"$a_{2}$",
            "phi_12": r"$\phi_{12}$",
            "phi_jl": r"$\phi_{\text{jl}}$"
            }
    return labels_dict

def hist_posterior(posterior_samples, x, outdir="./", parameters=None, labels_dict=None, density=False):
    """Plot a series of histograms showing how the training data varies to the posterior"""
    N = np.shape(x)[-1]
    fig, axs = plt.subplots(1,N, figsize=(24,4), sharey=True)
    axs = axs.ravel()
    n_max = np.shape(posterior_samples)[0]
    data_idx = np.random.permutation(range(np.shape(x)[0]))[:n_max]
    x1 = x[data_idx, :]
    x2 = posterior_samples
    for i in range(N):
        ax = axs[i]
        ax.hist(x1[:,i], 10, density=density, alpha=0.7,histtype="step", range=(0,1), hatch="-", linewidth=3.)
        ax.hist(x2[:,i], 10, density=density, alpha=0.7,histtype="step", range=(0,1), hatch="/", linewidth=3.)
        #ax.set_yscale("log")
        if parameters is not None:
            if labels_dict is not None:
                ax.set_xlabel(labels_dict[parameters[i]])
            else:
                ax.set_xlabel(parameters[i])
        #ax.set_yticklabels([])
        ax.set_xticks([0., 0.5, 1.0])
        axs[0].set_ylabel("Counts")
        ax.minorticks_on()
    fig.savefig(outdir + "data_hist.pdf", bbox_inches="tight")

def compare_run_to_posterior(run_path, sampling_results, use_training_results=False, outdir=None, fname="results.h5", plots=True, additional_metrics=None, scatter=True, blocks="all"):
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
        scatter: enable or disable scatter plots for each block
    Returns:
        data: list of tuples containing the predicted values and metrics dict

    """
    from gwfa.function_approximator import FunctionApproximator
    if run_path[-1] is not "/":
        run_path = run_path + "/"
    if outdir is None:
        outdir = run_path
    if not os.path.isdir(outdir):
        raise ValueError("Output directory does not exist")
    # format of posterior results determined by Bilby
    posterior_results = deepdish.io.load(sampling_results)["posterior"]
    posterior_values = posterior_results["logL"].values# + posterior_results["logPrior"].values
    # load the function approximator for this run
    FA = FunctionApproximator(attr_dict=run_path + "fa.pkl", verbose=0)
    # sort posterior according to order used in function approximator
    posterior_results = posterior_results.drop(["logL", "logPrior"], axis=1)[FA.parameter_names]
    posterior_samples = posterior_results.values
    parameters = list(posterior_results.columns.values)
    # get an ordered list of weights files
    weights_files = get_weights(run_path, weights_fname="model_weights.h5", blocks=blocks)
    data = []
    # load results from training if path provided:
    #if use_training_results:
    training_results = read_results(run_path, fname="results.h5", blocks=blocks, concat=False)
    for c, wf in enumerate(weights_files):
        print("Loading weights from: " +  wf)
        FA.load_weights(wf)
        # samples are normalised using the prior ranges that where used to set up FA
        normalised_samples, preds = FA.predict(posterior_samples)
        m = compare_to_posterior(posterior_samples, posterior_values, preds, additional_metrics)
        # scatter is slow to plot
        if scatter and plots:
            if use_training_results:
                x = [normalised_samples, training_results["x_val"][c]]
                y_true = [posterior_values, training_results["y_val"][c]]
                y_pred = [preds, training_results["y_pred"][c]]
            else:
                x = normalised_samples
                y_true = posterior_values
                y_pred = preds
            # make a scatter plot for the network after each block
            #grid_scatter(x, y_true, y_pred, outdir=outdir + "block{}/".format(c), parameters=parameters, fname="scatter_posterior.png", mkdir=True, labels_dict=get_labels_dict())
            hist_posterior(normalised_samples, training_results["x_val"][c], outdir=outdir+"block{}/".format(c), parameters=parameters, labels_dict=get_labels_dict())
        data.append((preds, m))

    if plots:
        print("Making plots")
        n_subplots = int(np.ceil(np.sqrt(len(weights_files))))
        hist_fig = plt.figure(figsize=(12, 10))
        metric_names = [k for k in data[0][1].keys()]
        metrics_array = np.empty([len(metric_names), c + 1])
        for i, d in enumerate(data):
            for j, name in enumerate(metric_names):
                metrics_array[j, i] = d[1][name]
            ax = hist_fig.add_subplot(n_subplots, n_subplots, i + 1)
            _, _, h1 = ax.hist(d[0], alpha=0.5, label="Predicted postreior", density=True)
            _, _, h2 = ax.hist(posterior_values, alpha=0.5, label="True posterior", density=True)
            ax.set_title("Block " + str(i))
            ax.set_xlabel("loglikelihood")
            ax.legend((blank_legend_entry(), blank_legend_entry()), ("KLD: {:.3f}".format(d[1]["KL"]), "JSD: {:.3f}".format(d[1]["JS"])))
        h, l = ax.get_legend_handles_labels()
        lgd = hist_fig.legend(h, l, loc ="lower center", fontsize=14)
        hist_fig.tight_layout()
        hist_fig.savefig(outdir + "hist.pdf", bbox_inches="tight")
        plt.close(hist_fig)

        # hist of last predictions
        hfig = plt.figure(figsize=(10,8))
        xmin = np.min([data[-1][0], posterior_values])
        xmax = np.max([data[-1][0], posterior_values])
        plt.hist(data[-1][0], 40, alpha=0.5, density=False,histtype="step", range=(xmin, xmax), hatch="-", linewidth=3.0)
        plt.hist(posterior_values, 40, alpha=0.5, density=False,histtype="step", range=(xmin, xmax), hatch="/", linewidth=3.0)
        plt.xlabel("Log-likelihood")
        plt.ylabel("Counts")
        plt.legend(["Predicted", "True"])
        hfig.tight_layout()
        hfig.savefig(outdir + "hist_final.pdf", bbox_inches="tight")



        metrics_fig = plt.figure(figsize=(12, 10))
        for i, m in enumerate(metrics_array):
            ax = metrics_fig.add_subplot(len(metrics_array), 1, i + 1)
            ax.plot(m, 'o')
            ax.set_yscale("log")
            ax.set_xlabel("Block")
            ax.set_ylabel(metric_names[i])
        metrics_fig.tight_layout()
        metrics_fig.savefig(outdir +  "metrics.png")
        plt.close(metrics_fig)
    if len(data) == 1:
        data = data[0]
    return data, FA

def search_for_runs(path):
    """Return a naturally sorted list of all the run directories in a given path"""
    dirs = filter(os.path.isdir, [path + i for i in os.listdir(path)])
    runs = filter(lambda d: "run" in d, dirs)
    runs_srt = sorted(runs, key=lambda s: int(s.split("run")[-1]))
    return runs_srt

def compare_search_posterior(results_dir, sampling_results, outdir=None, fname="results.h5", parameters=["neurons", "layers"], force_metrics=None):
    """Compare the runs in directory using the posterior samples"""
    if outdir is None:
        outdir = results_dir
    if not os.path.isdir(outdir):
        raise ValueError("Output directory does not exist")
    runs = search_for_runs(results_dir)
    metrics = np.empty(len(runs), dtype=np.ndarray)
    metric_names = False
    parameter_values = np.empty([len(runs), len(parameters)])
    for i, r in enumerate(runs):
        d, FA = compare_run_to_posterior(r, sampling_results, plots=False, blocks="last")
        metrics[i] = list(d[1].values())
        if not metric_names:
            metric_names = list(d[1].keys())
        parameter_values[i] = [FA.parameters[p] for p in parameters]
    # metrics is array of ndarrays, convert to a 2D array
    metrics = np.vstack(metrics).T
    n_metrics = metrics.shape[0]
    fig = plt.figure(figsize=(20, 8))
    cmap = mpl.cm.plasma_r(np.linspace(0,1,1000))
    cmap = mpl.colors.ListedColormap(cmap[100:,:-1])
    np.save("search_points.npy", parameter_values)
    np.save("metrics.npy", metrics)
    for i, m in enumerate(metrics):
        ax = fig.add_subplot(1, n_metrics, i + 1)
        sc = ax.scatter(*parameter_values.T, c=m, cmap=cmap, s=100)
        cbar = plt.colorbar(sc)
        cbar.set_label(metric_names[i])
        #ax.set_xlabel(parameters[0])
        ax.set_xlabel("Number of neurons")
        #ax.set_yticks(np.unique(parameter_values[:,1]))
        #ax.set_ylabel(parameters[1])
        ax.set_ylabel("Number of layers")
        #ax.set_xticks(np.unique(parameter_values.T[0, :]))
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=45)
    fig.tight_layout()
    fig.savefig(outdir + "metrics_posterior.png")
