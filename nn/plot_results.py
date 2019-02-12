#!/usr/bin/env python

import sys
from gwfa.utils.plotting import make_plots_multiple, compare_run_to_posterior

def main(posterior_samples, indir, outdir):
    """Make plots given some results"""
    compare_run_to_posterior(indir, posterior_samples, outdir=outdir, scatter=True)
    make_plots_multiple(indir, outdir, scatter=True, dd=True)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError("Missing posterior samples and input director")
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2], sys.argv[2])
    elif len(sys.argc) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        raise ValueError("Too many system arguments")

