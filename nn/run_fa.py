#!/usr/bin/env python

import sys
import time
import numpy as np

import six
from gwfa.data import Data
from gwfa.function_approximator import FunctionApproximator
from gwfa import utils


def get_model_path():
    """Get the model path from command line arguments"""
    if len(sys.argv) > 1:
        return sys.argv[1]
    else:
        return "model.json"

def main():
    model_path = get_model_path()
    params = utils.network.get_parameters_from_json(model_path)
    data = Data(params["datapath"])
    # priors should be an array of the parametesr in expected order
    priors_in, priors_ex = data.split_parameters()
    n_extrinsic = np.shape(priors_ex)[-1]
    n_intrinsic = np.shape(priors_in)[-1]
    if n_intrinsic == 0:
        input_shape = [n_extrinsic]
    elif n_extrinsic == 0:
        input_shape = [n_intrinsic]
    else:
        if params["split"]:
            # ordering is important if using split inputs
            print("\nTraining function approximator with split inputs\n")
            input_shape = [n_extrinsic, n_intrinsic]
        else:
            input_shape = sum([n_extrinsic, n_intrinsic])
    # the function approximator class should be able handle any of the previous input shapes
    FA = FunctionApproximator(input_shape, json_file=model_path, parameter_names=data.parameters)
    # FA will split inputs/priors automatically depending on how it was setup
    priors = np.concatenate([priors_ex, priors_in], axis=-1)
    FA.setup_normalisation(priors)
    data.prep_data_chain(block_size=params["block_size"], norm_logL=False, norm_intrinsic=False, norm_extrinsic=False)
    if params["blocks"] == "all":
        # avoid the first block with very large negative values of logL
        # -> start at 1
        if data.N_blocks == 1:
            blocks_2_train = range(0, 1)
        else:
            blocks_2_train = range(1, data.N_blocks)
    else:
        blocks_2_train = params["blocks"]
    print("Training on blocks: ", blocks_2_train)
    # input data need not be split but I wrote it like this originally don't see the need to change it
    # could be useful later
    for x_in, x_ex, y, i in zip(data.intrinsic_parameters, data.extrinsic_parameters, data.logL, range(data.N_blocks)):
        print(i)
        print(blocks_2_train)
        if i in blocks_2_train:
            # make sure the input data isn't split
            if x_ex.any() and x_in.any():
                x = np.concatenate([x_ex, x_in], axis=-1)
            elif x_ex.any():
                x = x_ex
            elif x_in.any():
                x = x_in
            FA.train_on_data(x, y, accumulate=params["accumulate"], plot=True, max_training_data=None, enable_tensorboard=False)
    if params["save"]:
        FA.save_results()
        FA.save_approximator("fa.pkl")

if __name__ == "__main__":
    main()
