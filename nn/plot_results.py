#!/usr/bin/env python

import sys
from gwfa.utils.plotting import make_plots_multiple, compare_run_to_posterior

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

d = {
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'text.usetex': True,
    'legend.fontsize': 'large',
    'figure.figsize': (12, 8),
    'axes.labelsize': 'xx-large',
    'axes.titlesize':'xx-large',
    'xtick.labelsize':'xx-large',
    'ytick.labelsize':'xx-large',
    'font.size': 28,
    "savefig.transparent": True,
    'font.family': 'Computer Modern',
    'font.monospace': 'Computer Modern Typewriter',
    'font.serif': 'Computer Modern Roman'
}


d1 = {
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'text.usetex': True,
    'legend.fontsize': 'large',
    'figure.figsize': (12, 8),
    'axes.labelsize': 'xx-large',
    'axes.titlesize':'xx-large',
    'font.size': 18,
    "savefig.transparent": True,
    'font.family': 'serif',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.labelsize':'xx-large',
    'ytick.labelsize':'xx-large',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 12.0,
    'ytick.major.size': 12.0,
    'xtick.minor.size': 6.0,
    'ytick.minor.size': 6.0,
    "font.weight": "bold",
    'text.latex.preamble': r"\usepackage{amsmath}"
}

plt.rcParams.update(d1)
plt.rcParams['hatch.linewidth'] = 2.0


def check_directory(d):
    """Check the directory path ends with '/'"""
    if d[-1] is not "/":
        d = d + "/"
    return d

def main(posterior_samples, indir, outdir):
    """Make plots given some results"""
    indir, outdir = check_directory(indir), check_directory(outdir)
    compare_run_to_posterior(indir, posterior_samples, outdir=outdir, scatter=True, blocks="all", use_training_results=False)
    #make_plots_multiple(indir, outdir, scatter=False, dd=True)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError("Missing posterior samples and input director")
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2], sys.argv[2])
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        raise ValueError("Too many system arguments")

