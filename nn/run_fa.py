import sys
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
    FA = FunctionApproximator(n_extrinsic, n_intrinsic, json_file=model_path, parameter_names=data.parameters)
    priors = np.concatenate([priors_ex, priors_in], axis=-1)
    FA.setup_normalisation(priors)
    data.prep_data_chain(block_size=params["block size"], norm_logL=False, norm_intrinsic=False, norm_extrinsic=False)
    if params["blocks"] == "all":
        blocks_2_train = range(1, data.N_blocks)
    else:
        blocks_2_train = params["blocks"]
    print("Training on blocks: ", blocks_2_train)

    for x_in, x_ex, y, i in zip(data.intrinsic_parameters, data.extrinsic_parameters, data.logL, range(data.N_blocks)):
        if i in blocks_2_train:
            if x_ex.any() and x_in.any():
                x = np.concatenate([x_ex, x_in], axis=-1)
            elif x_ex.any():
                x = x_ex
            elif x_in.any():
                x = x_in
            FA.train_on_data(x, y, accumulate=params["accumulate"], plot=True)

    FA.save_results()
    FA.save_approximator("fa.pkl")

if __name__ == "__main__":
    main()
