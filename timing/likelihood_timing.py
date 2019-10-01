#!/usr/bin/env python

import sys
import timeit
import numpy as np

# times to evaluate likelihood
number = int(1000)
if sys.argv[1] == "FA":
    path = "../nn/outdir/iota_psi_dist_ra_dec_m1_m2_marg_phase/run5/"
    setup_gpu="""\
import numpy as np
from utils import setup_function_approximator
FA = setup_function_approximator("{}", "gpu")
x = np.random.uniform(0, 1, (1,7))
""".format(path)
    setup_cpu = """\
import numpy as np
from utils import setup_function_approximator
FA = setup_function_approximator("{}", "cpu")
x = np.random.uniform(0, 1, (1,7))
""".format(path)
    run = """\
FA.model.predict(x)
"""
    if sys.argv[2] == "cpu":
        t = timeit.repeat(stmt=run, number=number, setup=setup_cpu, repeat=100)
        print("Function approximator with CPU ({}): {}s".format(number, t))
    else:
        t = timeit.repeat(stmt=run, number=number, setup=setup_gpu, repeat=100)
        print("Function approximator with GPU ({}): {}s".format(number, t))
    np.save("timeit_nn.npy", np.array(t))
elif sys.argv[1] == "bilby":
# time biliy likelihood
    setup="""\
import numpy as np
from utils import setup_bilby_likelihood
sampler, priors, parameters = setup_bilby_likelihood(8)
print("Sampling from {}D parameter space".format(len(parameters) - 1))
print("Parameters: {}".format(list(parameters.keys())))
"""
    run="""\
sampler.likelihood.log_likelihood()
"""
    t = timeit.repeat(stmt=run, number=number, setup=setup, repeat=10)
    print("Bilby ({}): {}s".format(number, t))
    np.save("timeit_times.npy", np.array(t))
