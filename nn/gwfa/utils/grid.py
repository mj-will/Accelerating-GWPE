
import sys
import os
import numpy as np
from sklearn.model_selection import ParameterGrid
from gwfa.utils.priors import get_prior_ranges



def likelihood_grid(priors, likelihood, N=10, outdir="./"):
    """
    Save a grid of likelihood points evenly sampled over the prior hypercube.
    """
    # get list of parameters and their ranges
    parameters, prior_ranges = get_prior_ranges(priors)
    # get dict to pass to sklearn function
    param_grid = {}
    fixed_params = {}
    for p, pr in zip(parameters, prior_ranges):
        if pr[0] == pr[1]:
            fixed_params[p] = pr[0]
        else:
            param_grid[p] = np.linspace(pr[0], pr[1], N)
    # grid
    grid = ParameterGrid(param_grid)
    # grid should be used as follows:
    # for parameters in grid:
    #     a_function(parameters['parameter1'], ...)
    M = len(grid)
    i = 0
    log_l = np.empty([M, 1])
    points = np.empty([M, len(param_grid)])
    for params in grid:
        # set the parameters
        likelihood.parameters = {**fixed_params, **params}
        points[i, :] = list(params.values())
        # compute logL
        log_l[i] = (likelihood.log_likelihood())
        if not i % 10:
            print(f"Progress: {i}/{M}", end="\r")
        i += 1
    print(f"Finished: {M}/{M}")
    sys.stdout.flush()
    print("Done")
    data = np.concatenate([points, log_l], axis=1)
    np.savetxt(os.path.join(
                outdir,"grid_samples.dat"),
                data,
                header=" ".join([*params] + ["logL"]),
                newline="\n",delimiter=" ")



