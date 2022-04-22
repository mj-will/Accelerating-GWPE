
import json
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, concatenate, Input, BatchNormalization
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras import regularizers

def KL(y_true, y_pred):
    """Return kullback-Leibler divergence"""
    y_true = K.exp(y_true)
    y_pred = K.exp(y_pred)
    P = (y_true / K.sum(y_true)) + K.epsilon()
    Q = y_pred / K.sum(y_pred) + K.epsilon()
    return K.sum(P * tf.math.log(P / Q))

def JSD(y_true, y_pred):
    """Compute the Jenson-Shannon divergence"""
    y_pred = K.exp(y_pred)
    y_true = K.exp(y_true)
    P = (y_true / K.sum(y_true)) + K.epsilon()
    Q = y_pred / K.sum(y_pred) + K.epsilon()
    const = K.constant(0.5)
    M = const * (P + Q)
    return const * K.sum(P * tf.math.log(P / M)) + const * K.sum(Q * tf.math.log(Q / M))

def get_parameters_from_json(model_path, verbose=1):
    """Get the parameters for the nn from the model.json file"""
    if verbose:
        print("Loading model: " + model_path)
    with open(model_path, "r") as read_file:
        params = json.load(read_file)
    return params

def network(n_inputs, parameters, verbose=1):
    """Get the model for neural network"""

    if type(n_inputs) == int:
        if n_inputs == 0:
            raise ValueError("Number of inputs must be non-zero")
        n_inputs = [n_inputs]
    else:
        if len(n_inputs) < 1 :
            raise ValueError("Must specifiy number of inputs")

    n_neurons = parameters["neurons"]
    try:
        n_mixed_neurons = parameters["mixed_neurons"]
    except:
        n_mixed_neurons = parameters["mixed neurons"]
    n_layers = parameters["layers"]
    try:
        n_mixed_layers = parameters["mixed_layers"]
    except:
        n_mixed_layers = parameters["mixed layers"]
    dropout_rate = parameters["dropout"]
    try:
        mixed_dropout_rate = parameters["mixed_dropout"]
    except:
        mixed_dropout_rate = parameters["mixed dropout"]
    try:
        batch_norm = parameters["batch_norm"]
    except:
        batch_norm = parameters["batch norm"]
    activation = parameters["activation"]
    regularization = parameters["regularization"]

    if not isinstance(n_neurons, (list, tuple, np.ndarray)):
        n_neurons = n_neurons * np.ones(n_layers, dtype=int)
    if not isinstance(n_mixed_neurons, (list, tuple, np.ndarray)):
        n_mixed_neurons = n_mixed_neurons * np.ones(n_mixed_layers, dtype=int)
    if not len(n_neurons) is n_layers:
        raise ValueError("Specified more layers than neurons")

    if activation == "erf":
        activation = tf.erf
    elif activation == "probit":
        def probit(x):
            """return probit of x"""
            normal = tf.distributions.Normal(loc=0., scale=1.)
            return normal.cdf(x)
        actiavtion = probit

    if regularization == "l1":
        reg = regularizers.l1(parameters["lambda"])
    elif regularization == "l2":
        reg = regularizers.l2(parameters["lambda"])
    else:
        print("Proceeding with no regularization")
        reg = None

    inputs = []
    block_outputs = []
    for i, n in enumerate(n_inputs):
        layer = Input(shape=(n,), name="input_" + str(i + 1))
        inputs.append(layer)
        for j in range(n_layers):
            layer = Dense(n_neurons[i], activation=activation, kernel_regularizer=reg, name="p{}_dense_{}".format(i, j))(layer)
            if dropout_rate:
                layer = Dropout(dropout_rate)(layer)
            if batch_norm:
                layer = BatchNormalization()(layer)
        block_outputs.append(layer)
    if len(block_outputs) > 1:
        layer = concatenate(block_outputs, name="concat_blocks")
        for i in range(n_mixed_layers):
            layer = Dense(n_mixed_neurons[i], activation=activation, kernel_regularizer=reg, name="mix_dense_{}".format(i))(layer)
            if mixed_dropout_rate:
                layer = Dropout(mixed_dropout_rate)(layer)
            if batch_norm:
                outputs = BatchNormalization()(layer)
    # make final layer
    output_layer = Dense(1, activation="linear", name="output_dense")(layer)
    model = Model(inputs=inputs, outputs=output_layer)
    if verbose:
        model.summary()

    return model
