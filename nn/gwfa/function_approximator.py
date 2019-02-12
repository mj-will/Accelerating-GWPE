
import os
import shutil
import six
import numpy as np
import deepdish

from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
import tensorflow as tf
import keras.backend as K

from gwfa import utils

# TensorFlow wizardry
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))

class FunctionApproximator(object):

    def __init__(self, n_extrinsic=None, n_intrinsic=None, parameter_names=None, json_file=None, attr_dict=None):
        if json_file is not None and attr_dict is not None:
            raise ValueError("Provided both json file and attribute dict, use one or other")
        elif json_file is not None:
            self.n_extrinsic = n_extrinsic
            self.n_intrinsic = n_intrinsic
            self.compiled = False
            self.model = None
            self._count = 0
            self.normalise = False
            if parameter_names is None:
                self.parameter_names = ["parameter_" + str(i) for i in range(n_extrinsic + n_intrinsic)]
            else:
                self.parameter_names = parameter_names
            self.setup_from_json(json_file)
        elif attr_dict is not None:
            self.setup_from_attr_dict(attr_dict)
        else:
            raise ValueError("No json file or saved FA file for setup")

    def __str__(self):
        args = "".join("{}: {}\n".format(key, value) for key, value in six.iteritems(self.parameters))
        return "FunctionApproximator instance\n" + "".join("    {}: {}\n".format(key, value) for key, value in six.iteritems(self.parameters))

    def setup_from_attr_dict(self, attr_dict):
        """Set up the approximator from a dictionary of attributes"""
        with open(attr_dict, "rb") as f:
            d = six.moves.cPickle.load(f)
        for key, value in six.iteritems(d):
            setattr(self, key, value)
        self.model = utils.network.network(self.n_extrinsic, self.n_intrinsic, self.parameters)
        self._compile_network()

    def setup_from_json(self, json_file):
        """Set up the class before training from a json file"""
        if self.model is None:
            print("Setting up function approximator")
            self.parameters = utils.network.get_parameters_from_json(json_file)
            self.outdir = self.parameters["outdir"]
            self._setup_directories()
            shutil.copy(json_file, self.tmp_outdir)
            # setup network
            self.model = utils.network.network(self.n_extrinsic, self.n_intrinsic, self.parameters)
            self._compile_network()
            self.data_all = {}
        else:
            print("Function approximator already setup")

    def _setup_directories(self):
        """Setup final output directory and a temporary directory for use during training"""
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
        self.tmp_outdir = './current_run/'
        # directory may still exist from previous runs that did not terminate properly
        # delete to prevents results from getting mixed up
        if os.path.exists(self.tmp_outdir):
            shutil.rmtree(self.tmp_outdir)
        os.mkdir(self.tmp_outdir)

    def _compile_network(self):
        """Setup the loss function and compile the model"""
        if self.parameters["loss"] == "KL":
            print("KL divergence")
            self.model.compile(Adam(lr=self.parameters["learning_rate"], decay=self.parameters["lr_decay"]), loss=utils.network.KL, metrics=["mse"])
        else:
            print("Using " + self.parameters["loss"])
            self.model.compile(Adam(lr=self.parameters["learning_rate"], decay=self.parameters["lr_decay"]), loss=self.parameters["loss"], metrics=[utils.network.KL])

        self.compiled = True

    def _shuffle_data(self, x, y):
        """Shuffle data"""
        p = np.random.permutation(len(y))
        return x[p], y[p]

    def setup_normalisation(self, priors):
        """
        Get range of priors for the parameters to be later used for normalisation
        NOTE: expect parameters to be ordered: extrinsic, intrinsic
        """
        self._prior_max = np.max(priors, axis=0)
        self._prior_min = np.min(priors, axis=0)
        self.normalise = True

    def _normalise_input_data(self, x):
        """Normalise the input data given the prior values provided at setup"""
        return (x - self._prior_min) / (self._prior_max - self._prior_min)

    @property
    def _training_parameters(self):
        """Return a dictionary of the parameters to be passed to model.fit"""
        return {"epochs": self.parameters["epochs"], "batch_size": self.parameters["batch_size"]}

    def _split_data(self, x):
        """Split data according to number of extrinsic and intrinsic parameters"""
        if self.n_extrinsic and self.n_intrinsic:
            x_split = [x[:, :self.n_extrinsic], x[:, -self.n_intrinsic:]]
        else:
            x_split = x
        return x_split

    def train_on_data(self, x, y, split=0.8, accumulate=False, plot=False):
        """
        Train on provided data

        Args:
            x : list of array-like samples
            y : list of true values
        """
        if not self.compiled:
            raise RuntimeError("Model must be compiled before training")
        block_outdir = self.tmp_outdir + "block{}/".format(self._count)
        if not os.path.isdir(block_outdir):
            os.mkdir(block_outdir)

        if self.normalise:
            x = self._normalise_input_data(x)
        # accumlate data if flag true and not the first instance of training
        if accumulate and self._count:
            x = np.concatenate([self._accumulated_data[0], x], axis=0)
            y = np.concatenate([self._accumulated_data[1], y], axis=0)

        x, y = self._shuffle_data(x, y)
        n = len(y)
        x_train, x_val = np.array_split(x, [int(split * n)], axis=0)
        self.y_train, self.y_val = np.array_split(y, [int(split * n)], axis=0)
        self.x_train = self._split_data(x_train)
        self.x_val = self._split_data(x_val)
        if accumulate is not False:
            if accumulate == "val":
                self._accumulated_data = (x_val, self.y_val)
            elif accumulate == "train":
                self._accumulated_data = (x_train, self.y_train)
            elif accumulate == "all":
                acc_x = np.concatenate([x_train, x_val], axis=0)
                acc_y = np.concatenate([self.y_train, self.y_val], axis=0)
                self._accumulated_data = (acc_x, acc_y)
            else:
                raise ValueError("Unknown data type to accumulate: {}. Choose from: 'val', 'train' or 'all'".format(accumulate))

        callbacks = []
        if self.parameters["patience"]:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=self.parameters["patience"]))
        # save best model during traing
        self.weights_file = block_outdir + "model_weights.h5"
        checkpoint = ModelCheckpoint(self.weights_file, verbose=0, monitor="val_loss", save_best_only=True, mode="auto")
        callbacks.append(checkpoint)
        # more callbacks can be added by appending to callbacks
        history = self.model.fit(x=self.x_train, y=self.y_train, validation_data=(self.x_val, self.y_val), verbose=2, callbacks=callbacks, **self._training_parameters)
        y_train_pred = self.model.predict(self.x_train)
        # load weights from best epoch
        self.model.load_weights(self.weights_file)
        y_pred = self.model.predict(self.x_val).ravel()
        # save the x arrays before they're split into extrinsic/intrinsic
        results_dict = {"x_train": x_train,
                        "x_val": x_val,
                        "y_train": self.y_train,
                        "y_val": self.y_val,
                        "y_train_pred": y_train_pred,
                        "y_pred": y_pred,
                        "parameters": self.parameter_names}
        results_dict.update(history.history)
        if plot:
            utils.plotting.make_plots(block_outdir, **results_dict)
        self.data_all["block{}".format(self._count)] = results_dict
        self._count += 1

    def load_weights(self, weights_file):
        """Load weights for the model"""
        self.model.load_weights(weights_file)

    def predict(self, x):
        """Get predictions for a given set of points in parameter space that have not been normalised"""
        x = self._normalise_input_data(x)
        x = self._split_data(x)
        y = self.model.predict(x).ravel()
        return x, y

    def _make_run_dir(self):
        """Check run count and make outdir"""
        run = 0
        while os.path.isdir(self.outdir + 'run{}'.format(run)):
            run += 1

        run_path = self.outdir + 'run{}/'.format(run)
        if not os.path.exists(run_path):
            os.mkdir(run_path)
        self._run_path = run_path
        return run_path

    def save_results(self, save=False):
        """Save the results from the complete training process and move to final save directory"""
        if self.parameters["save"] or save:
            self.run_outdir = self._make_run_dir()
            deepdish.io.save(self.run_outdir + "results.h5", self.data_all)
            utils.copytree(self.tmp_outdir, self.run_outdir)
            # empty current dir
            for f in os.listdir(self.tmp_outdir):
                file_path = os.path.join(self.tmp_outdir, f)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
        else:
            print("JSON file parameters specified not to save data, skipping. To force saving enable it in this function (not recommeneded)")

    def save_approximator(self, fname="fa.pkl"):
        """Save the attributes of the function approximator"""
        print("Saving approximator as a dictionary of attributes")
        attr_dict = vars(self)
        attr_dict.pop("model")
        with open(self._run_path + fname, "wb") as f:
            six.moves.cPickle.dump(attr_dict, f, protocol=six.moves.cPickle.HIGHEST_PROTOCOL)
