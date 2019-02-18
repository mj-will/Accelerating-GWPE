#!/usr/bin/env python

import sys
from gwfa.utils.plotting import compare_runs_2d, compare_search_posterior

def main(posterior_samples, indir, outdir):
    """Make plots given some results"""
    compare_runs_2d(indir, outdir, run_start=7, block="last")
    compare_search_posterior(indir, posterior_samples)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError("Missing posterior samples and input director")
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2], sys.argv[2])
    elif len(sys.argc) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        raise ValueError("Too many system arguments")

