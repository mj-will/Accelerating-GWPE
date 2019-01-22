# test of basic nn to approximate a loglikelihood

from __future__ import print_function

import shutil
import os
import json
import numpy as np
import pandas as pd
import h5py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, concatenate, Input
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import regularizers
from keras.losses import kullback_leibler_divergence

import tensorflow as tf

from scipy.special import expit as sigmoid

from utils import *

# TensorFlow wizardry
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))

def shuffle_data(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def shuffle_data_3d(a, b, c):
    assert len(a) == len(b)
    assert len(a) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]

class Data(object):

    def __init__(self, data_path=None, ignore=None):
        """Initialize a class to load and process data"""
        if not data_path is None:
            self.data_path = data_path
            self.samples_path = data_path + "nested_samples.dat"
            self.header_path = data_path + "header.txt"
            self.ignore = ignore
            self.df = None

            if not os.path.isfile(self.samples_path):
                raise ValueError("Could not find 'nested_samples.dat' in given data directory:" + self.samples_path)

            if not os.path.isfile(self.header_path):
                raise ValueError("Could not find 'header.txt' in given data directory:" + self.header_path)

            self._load_data(ignore=self.ignore)

    @property
    def parameters(self):
        """Return a list of parameter names in the data"""
        if self.df is None:
            raise RuntimeError("Dataframe not loaded")
            return None
        else:
            return self.df.columns.values[:-2]    # last two columns are logL and logPrior

    @property
    def _extrinsic_parameters(self):
        """Return a list of extrinsic parameters"""
        return list(['ra', 'dec', 'iota', 'psi', 'luminosity_distance', 'phase', 'geocent_time'])

    @property
    def _intrinsic_parameters(self):
        """Return a list on intrinsic parameters"""
        return list(['m1', 'm2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jli', 'mass_1', 'mass_2'])

    @property
    def _header_list(self):
        """Return a list of the headers"""
        header = list(np.genfromtxt(self.header_path, dtype=str))
        # make sure logPrior is included
        if header[-1] is not 'logPrior':
            header.append('logPrior')
        return list(header)

    def _load_data(self, ignore = None):
        """Load the data using pandas"""
        self.df = pd.read_csv(self.samples_path, sep=" ", header=0, names=self._header_list, escapechar='#')
        if ignore is not None:
            print(ignore)
            self._orig_df = self.df
            self.df = self.df.drop(ignore, axis=1)

        #self.df = self.df.ix[10000:]
        self.print_data()

    def print_data(self):
        """Print a summary of the data"""
        print('Columns in data:')
        for name, values in self.df.iteritems():
            print('Name: {:20s} Min: {:9.5f} Max: {:9.5f} Mean: {:9.5f}'.format(name, np.min(values), np.max(values), np.mean(values)))

    def split_parameters(self):
        """Return a list of the extrinsic and intrinsic parameters in the data"""
        if self.df is None:
            raise ValueError('Dataframe not loaded')

        self._intrinsic_names = [name for name in self.df.columns.values if name in self._intrinsic_parameters]
        self._extrinsic_names = [name for name in self.df.columns.values if name in self._extrinsic_parameters]
        print('Intrinsic parameters in data: ', self._intrinsic_names)
        print('Extrinsic parameters in data: ', self._extrinsic_names)
        # get values for parameters from df
        intrinsic_parameters = self.df[self._intrinsic_names].values
        extrinsic_parameters = self.df[self._extrinsic_names].values
        return intrinsic_parameters, extrinsic_parameters

    def normalize_parameters(self, x):
        """Normalize the input parameters"""
        return (x - x.min()) / (x.max() - x.min())

    def normalize_logL(self, x):
        """Normalize the logL"""
        return (x - x.min()) / (x.max() - x.min())

    def prep_data_chain(self, block_size=1000, norm_logL=True, norm_intrinsic=False, norm_extrinsic=False):
        """
        Prep the data to emulate the output of a nested sampling algorithm
        """
        self.block_size = block_size
        self.logL = np.nan_to_num(np.squeeze(self.df[['logL']].values))
        self.N_points = len(self.logL)
        self.intrinsic_parameters, self.extrinsic_parameters = self.split_parameters()
        print('Number of data points:', self.N_points)
        self.N_intrinsic = self.intrinsic_parameters.shape[-1]
        self.N_extrinsic = self.extrinsic_parameters.shape[-1]
        # split the data in fragments that are the block size
        # get number of blocks
        self.N_blocks = int(np.floor(self.N_points / self.block_size))
        print('Block size:', self.block_size)
        print('Number of blocks:', self.N_blocks)
        diff = int(np.abs(self.N_blocks * block_size - self.N_points))
        # drop inital values
        if norm_logL:
            self.logL = np.apply_along_axis(self.normalize_logL, 1, (self.logL[:-diff].reshape(self.N_blocks, self.block_size)))
        else:
            self.logL = self.logL[:-diff].reshape(self.N_blocks, self.block_size)
        # intrinsic
        if self.N_intrinsic:
            if norm_intrinsic:
                self.intrinsic_parameters = np.apply_along_axis(self.normalize_parameters, 1, (self.intrinsic_parameters[:-diff, :].reshape(self.N_blocks, self.block_size, self.N_intrinsic)))
            else:
                self.intrinsic_parameters = self.intrinsic_parameters[:-diff, :].reshape(self.N_blocks, self.block_size, self.N_intrinsic)
        # extrinsic
        if self.N_extrinsic:
            if norm_extrinsic:
                self.extrinsic_parameters = np.apply_along_axis(self.normalize_parameters, 1, (self.extrinsic_parameters[:-diff, :].reshape(self.N_blocks, self.block_size, self.N_extrinsic)))
            else:
                self.extrinsic_parameters = self.extrinsic_parameters[:-diff, :].reshape(self.N_blocks, self.block_size, self.N_extrinsic)
        print('X shape:', self.intrinsic_parameters.shape, self.extrinsic_parameters.shape)
        print('Y shape:', self.logL.shape)


def network(N_intrinsic, N_extrinsic, params):
    """Get the model for neural network"""
    N_neurons = params['neurons']
    N_mixed_neurons = params['mixed neurons']
    N_layers = params['layers']
    N_mixed_layers = params['mixed layers']
    dropout = params['dropout']
    activation = params['activation']

    def probit(x):
        """return probit of x"""
        normal = tf.distributions.Normal(loc=0., scale=1.)
        return normal.cdf(x)

    if activation == 'erf':
        activation = tf.erf
    elif actiavtion == 'probit':
        actiavtion = probit

    if not N_intrinsic and not N_extrinsic:
        raise ValueError('Specified no intrinsic or extrinsic parameters. Cannot make a network with no inputs!')

    if not isinstance(N_neurons, (list, tuple, np.ndarray)):
        N_neurons = N_neurons * np.ones(N_layers, dtype=int)
    if not isinstance(N_mixed_neurons, (list, tuple, np.ndarray)):
        N_mixed_neurons = N_mixed_neurons * np.ones(N_mixed_layers, dtype=int)
    if not len(N_neurons) is N_layers:
        raise ValueError('Specified more layers than neurons')
    inputs = []
    output_layers = []
    if N_intrinsic:
        IN_input = Input(shape=(N_intrinsic,), name='intrinsic_input')
        inputs.append(IN_input)
        for i in range(N_layers):
            if i is 0:
                IN = Dense(N_neurons[i], activation=activation, name='intrinsic_dense_{}'.format(i))(IN_input)
            else:
                IN = Dense(N_neurons[i], activation=activation, name='intrinsic_dense_{}'.format(i))(IN)
        output_layers.append(IN)
    if N_extrinsic:
        EX_input = Input(shape=(N_extrinsic,), name='extrinsic_input')
        inputs.append(EX_input)
        for i in range(N_layers):
            if i is 0:
                EX = Dense(N_neurons[i], activation=activation, name='extrinsic_dense_{}'.format(i))(EX_input)
            else:
                EX = Dense(N_neurons[i], activation=activation, name='extrinsic_dense_{}'.format(i))(EX)
        output_layers.append(EX)
    # make model
    if len(output_layers) > 1:
        outputs = concatenate(output_layers, name='merge_intrinsic_extrinsic')
    else:
        outputs = output_layers[-1]
    # add mixed layers:
    for i in range(N_mixed_layers):
        outputs = Dense(N_mixed_neurons[i], activation=activation, name='mixed_dense_{}'.format(i))(outputs)
    # make final layer
    output_layer = Dense(1, activation='linear', name='output_dense')(outputs)
    model = Model(inputs=inputs, outputs=output_layer)
    # print model
    model.summary()
    return model

def custom_loss(a, sigma, r=0.01):
    """loss function used in BAMBI paper"""
    M = tf.cast(tf.size(a), tf.float32)    # total number of parameters
    sigma = tf.cast(sigma, tf.float32)     # 'variance' parameter
    pi = tf.cast(np.pi, tf.float32)        # pi with correct dtype for tf
    def loss_function(y_true, y_pred):
        loss = 0.
        K = tf.cast(tf.size(y_true), tf.float32)
        logLikelihood = lambda y_true, y_pred : - (K * tf.log(2. * pi)) / 2. - tf.log(sigma) - ((1. / 2.) * tf.reduce_sum(((y_true - y_pred) / sigma) ** 2.))
        logL = logLikelihood(y_true, y_pred)
        alpha = tf.norm(tf.gradients(logL, [y_pred])) / tf.sqrt(M) * r
        logS = - alpha / 2. * tf.reduce_sum(a ** 2.)
        loss = logL + logS
        return -loss

    return loss_function

def KL(y_true, y_pred):
    """Return kullback-Leibler divergence"""
    P = (y_true / tf.reduce_sum(y_true))
    Q = (y_pred / tf.reduce_sum(y_pred))
    return kullback_leibler_divergence(P,Q)

def train_approximator(X_IN, X_EX, Y, model, block_number=0, outdir = './', schedule=None, parameters=None, **kwargs):
    block_outdir = outdir + 'block{}/'.format(block_number)
    if not os.path.isdir(block_outdir):
        os.mkdir(block_outdir)
    # setup up training and validation data
    # depends on what parameters are in data (intrinsic and/or extrinsic)
    # 1. shuffle data
    # 2. split into train/val
    print('Training on block ', block_number)
    mixed_flag = False
    if not len(X_EX):
        # only intrinsic
        X_IN, Y = shuffle_data(X_IN, Y)
        X_IN_train, X_IN_val = np.array_split(X_IN, [int(0.8 * X_IN.shape[0])], axis=0)
        X_train = X_IN_train
        X_val = X_IN_val
    elif not len(X_IN):
        # only extrinsic
        X_EX, Y = shuffle_data(X_EX, Y)
        X_EX_train, X_EX_val = np.array_split(X_EX, [int(0.8 * X_EX.shape[0])], axis=0)
        X_train = X_EX_train
        X_val = X_EX_val
    else:
        # mix
        mixed_flag = True
        X_IN, X_EX, Y = shuffle_data_3d(X_IN, X_EX, Y)
        X_IN_train, X_IN_val = np.array_split(X_IN, [int(0.8 * X_IN.shape[0])], axis=0)
        X_EX_train, X_EX_val = np.array_split(X_EX, [int(0.8 * X_EX.shape[0])], axis=0)
        X_train = [X_IN_train, X_EX_train]
        X_val = [X_IN_val, X_EX_val]
    # logL doesn't depend on mix of intrinsic and/or extrinsic
    Y_train, Y_val = np.array_split(Y, [int(0.8 * Y.shape[0])], axis=0)

    # call backs
    callbacks = []
    # use schedule if provided
    if schedule is not None:
        LRS = LearningRateScheduler(schedule=schedule)
        callbacks.append(LRS)
    # save best model during traing
    checkpoint = ModelCheckpoint(block_outdir + 'model.h5', verbose=0, monitor='val_loss', save_best_only=True, mode='auto')
    callbacks.append(checkpoint)
    # more callbacks can be added by appending to callbacks
    # fit
    history = model.fit(x=X_train, y=Y_train, validation_data=(X_val, Y_val),
                        verbose=2, callbacks=callbacks, **kwargs)
    # load best weights
    model.load_weights(block_outdir + 'model.h5')
    # delete model file
    os.remove(block_outdir + 'model.h5')
    # evaluate best epoch
    eval_loss = model.evaluate(x=X_val, y=Y_val, verbose=2)
    print('Validation loss (best epoch):', eval_loss)
    # predictions
    preds = np.squeeze(model.predict(X_val))
    training_output = model.predict(X_train)
    # plot loss
    block_outdir = outdir + 'block{}/'.format(block_number)
    if not os.path.isdir(block_outdir):
        os.mkdir(block_outdir)
    #########
    # plots #
    #########
    if mixed_flag:
        x = np.concatenate(X_val, axis=0)
    else:
        x = X_val
    make_plots(block_outdir, x=x, y_true=Y_val, y_pred=preds, y_train_true=Y_train, y_train_pred=training_output, history=history, parameters=list(parameters))
    # save results
    hf = h5py.File(outdir + 'results.h5', 'a')
    grp = hf.create_group('block{}'.format(block_number))
    grp.create_dataset('y_true', data=Y_val)
    grp.create_dataset('y_pred', data=preds)
    grp.create_dataset('x', data=x)
    grp.create_dataset('parameters', data=list(parameters))
    hf.close()

    return history, model

def get_schedule(lr):
    """Return function that returns lr given an epoch"""
    def schedule(epoch):
        return lr *  (1. - epoch * 1e-4)
    return schedule

def get_model_params():
    """get a tf array of all the trainable parameters in the network"""
    variables_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    params = tf.concat([tf.reshape(v, [-1]) for v in variables_list], axis=0, name='model_params')
    return params

def get_network_params_from_json():
    """get the parameters for the nn from the model.json file"""
    with open("model.json", "r") as read_file:
        params = json.load(read_file)
    return params

def main():
    # main paths
    # get directory for results to be saved in

    network_params = get_network_params_from_json()
    data_path = network_params['datapath']
    outdir = network_params['outdir']

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # directory to store results during run
    tmp_outdir = './current_run/'
    if os.path.exists(tmp_outdir):
        shutil.rmtree(tmp_outdir)
    os.mkdir(tmp_outdir)
    # copy json file for run
    shutil.copy('model.json', tmp_outdir)

    # init class with data
    data = Data(data_path)#, ignore=['geocent_time'])
    # prep data
    train_blocks = True
    # starting training from previous block
    transfer=True
    if train_blocks:
        data.prep_data_chain(block_size=network_params['block size'], norm_logL=False, norm_intrinsic=True, norm_extrinsic=True)
        if network_params['blocks'] == 'all':
            blocks_2_train = range(1, data.N_blocks)
        else:
            blocks_2_train = network_params['blocks']
        print('Training on blocks: ', blocks_2_train)
    ##################
    # Neural Network #
    ##################

    # make model
    model = network(data.N_intrinsic, data.N_extrinsic, network_params)
    # compile model
    if network_params['loss'] == "bambi":
        print('Using BAMBI loss')
        # get weights and biases for alternate loss function
        model_params = get_model_params()
        model.compile(Adam(lr=network_params['learning rate']), loss=custom_loss(model_params, network_params['sigma'], r=network_params['r']), metrics=['mse'])
    elif network_params['loss'] == 'KL':
        print('KL divergence')
        model.compile(Adam(lr=network_params['learning rate']), loss=KL, metrics=['mse', KL])
    else:
        print('Using' + network_params['loss'])
        model.compile(Adam(lr=network_params['learning rate'], decay=network_params['lr decay']), loss=network_params['loss'], metrics=[KL])
    schedule = get_schedule(network_params['learning rate'])

    if train_blocks:
        # flag for training run
        trained=False
        for x_in, x_ex, y, i in zip(data.intrinsic_parameters, data.extrinsic_parameters, data.logL, range(data.N_blocks)):
            if i not in blocks_2_train:
                continue
            elif not trained:
                history, updated_model = train_approximator(x_in, x_ex, y, model, block_number=i, outdir=tmp_outdir, parameters=data.parameters, epochs=network_params['epochs'], batch_size=network_params['batch size'])
                if transfer:
                    trained=True
            else:
                history, updated_model_ = train_approximator(x_in, x_ex, y, updated_model, block_number=i, outdir=tmp_outdir, parameters=data.parameters, epochs=network_params['epochs'], batch_size=network_params['batch size'])
            current_lr = float(K.get_value(model.optimizer.lr))
            print('Learning rate after block {}: {}'.format(i, current_lr))
            #K.set_value(model.optimizer.lr, network_params['learning rate'])

    # once finished copy results to correct localtion
    if network_params["save"]:
        run_outdir = make_run_dir(outdir)
        copytree(tmp_outdir, run_outdir)
        # empty current dir
        for f in os.listdir(tmp_outdir):
            file_path = os.path.join(tmp_outdir, f)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

if __name__ == '__main__':
    main()
