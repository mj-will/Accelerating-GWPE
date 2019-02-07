
from __future__ import print_function

import json
import numpy as np
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, concatenate, Input
from keras.layers.normalization import BatchNormalization
from keras import regularizers

def KL(y_true, y_pred):
    """Return kullback-Leibler divergence"""
    return tf.reduce_sum(y_true * tf.log(y_true / y_pred))

def get_parameters_from_json(model_path, verbose=1):
    """Get the parameters for the nn from the model.json file"""
    if verbose:
        print("Loading model: " + model_path)
    with open(model_path, "r") as read_file:
        params = json.load(read_file)
    return params

def get_predictions(x, model, N_intrinsic, N_extrinsic, weights_file=None):
    """Get prediction froma model or model file

    Args:
        x: Values to evaluate
        model: An instance of a keras model or a string pointing to a .json model
        N_intrinsic: Number of intrinsic parameters
        N_extrinsic: Number of extrinsic parameters

    """
    params = False
    if type(model) == str:
        if ".json" in model:
            params = get_parameters_from_json(model)
            model = network(N_intrinsic, N_extrinsic, params)
        else:
            raise ValueError("Trying to load model from unknown format")
    elif type(model) == keras.models.Model:
        pass
    else:
        raise ValueError("Unknown model format")
    # load weights if file provided
    # need to in load weights for a block, but model hasn"t changed
    if weights_file is not None:
        model.load_weights(weights_file)
    elif params:
        print("Built model from json but could not load weights")

    return model.predict(x), model


def network(N_extrinsic, N_intrinsic, params):
    """Get the model for neural network"""
    N_neurons = params["neurons"]
    N_mixed_neurons = params["mixed neurons"]
    N_layers = params["layers"]
    N_mixed_layers = params["mixed layers"]
    dropout = params["dropout"]
    mixed_dropout = params["mixed dropout"]
    bn = params["batch norm"]
    activation = params["activation"]
    regularization = params["regularization"]

    if not N_intrinsic and not N_extrinsic:
        raise ValueError("Specified no intrinsic or extrinsic parameters. Cannot make a network with no inputs!")

    if not isinstance(N_neurons, (list, tuple, np.ndarray)):
        N_neurons = N_neurons * np.ones(N_layers, dtype=int)
    if not isinstance(N_mixed_neurons, (list, tuple, np.ndarray)):
        N_mixed_neurons = N_mixed_neurons * np.ones(N_mixed_layers, dtype=int)
    if not len(N_neurons) is N_layers:
        raise ValueError("Specified more layers than neurons")

    def probit(x):
        """return probit of x"""
        normal = tf.distributions.Normal(loc=0., scale=1.)
        return normal.cdf(x)

    if activation == "erf":
        activation = tf.erf
    elif activation == "probit":
        actiavtion = probit

    if regularization == "l1":
        reg = regularizers.l1(params["lambda"])
    elif regularization == "l2":
        reg = regularizers.l2(params["lambda"])
    else:
        print("Proceeding with no regularization")
        reg = None

    inputs = []
    output_layers = []
    if N_extrinsic:
        EX_input = Input(shape=(N_extrinsic,), name="extrinsic_input")
        inputs.append(EX_input)
        for i in range(N_layers):
            if i is 0:
                EX = Dense(N_neurons[i], activation=activation, kernel_regularizer=reg, name="extrinsic_dense_{}".format(i))(EX_input)
            else:
                EX = Dense(N_neurons[i], activation=activation, kernel_regularizer=reg, name="extrinsic_dense_{}".format(i))(EX)
                if dropout:
                    EX = Dropout(dropout)(EX)
                if bn:
                    EX = BatchNormalization()(EX)
        output_layers.append(EX)
    if N_intrinsic:
        IN_input = Input(shape=(N_intrinsic,), name="intrinsic_input")
        inputs.append(IN_input)
        for i in range(N_layers):
            if i is 0:
                IN = Dense(N_neurons[i], activation=activation, kernel_regularizer=reg, name="intrinsic_dense_{}".format(i))(IN_input)
            else:
                IN = Dense(N_neurons[i], activation=activation,  kernel_regularizer=reg, name="intrinsic_dense_{}".format(i))(IN)
                if dropout:
                    IN = Dropout(dropout)(IN)
                if bn:
                    IN = BatchNormalization()(IN)
        output_layers.append(IN)
    # make model
    if len(output_layers) > 1:
        outputs = concatenate(output_layers, name="merge_intrinsic_extrinsic")
    else:
        outputs = output_layers[-1]
    # add mixed layers:
    for i in range(N_mixed_layers):
        outputs = Dense(N_mixed_neurons[i], activation=activation, kernel_regularizer=reg, name="mixed_dense_{}".format(i))(outputs)
        if mixed_dropout:
            outputs = Dropout(mixed_dropout)(outputs)
        if bn:
            outputs = BatchNormalization()(outputs)
    # make final layer
    output_layer = Dense(1, activation="linear", name="output_dense")(outputs)
    model = Model(inputs=inputs, outputs=output_layer)
    # print model
    model.summary()
    return model
