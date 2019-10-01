#!/usr/bin/env python

import sys
from gwfa.utils.plotting import compare_runs_2d, compare_search_posterior

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

d1 = {
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'text.usetex': True,
    'legend.fontsize': 'large',
    'figure.figsize': (12, 8),
    'axes.labelsize': 'x-large',
    'axes.titlesize':'x-large',
    'font.size': 18,
    "savefig.transparent": True,
    'font.family': 'serif',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.labelsize':'x-large',
    'ytick.labelsize':'x-large',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 12.0,
    'ytick.major.size': 12.0,
    'xtick.minor.size': 6.0,
    'ytick.minor.size': 6.0,
    "font.weight": "bold"
}

plt.rcParams.update(d1)


def main(posterior_samples, indir, outdir):
    """Make plots given some results"""
    #compare_runs_2d(indir, outdir, block="all")
    compare_search_posterior(indir, posterior_samples)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError("Missing posterior samples and input director")
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2], sys.argv[2])
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        raise ValueError("Too many system arguments")

