
import os
import shutil
import six
import numpy as np
import deepdish

from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping

import utils

class FunctionApproximator(object):

    def __init__(self, n_extrinsic, n_intrinsic, parameter_names=None, json_file=None):
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
        if json_file is not None:
            self.setup_from_json(json_file)
        else:
            raise ValueError("No json file for setup")

    def __str__(self):
        args = "".join("{}: {}\n".format(key, value) for key, value in six.iteritems(self.parameters))
        #print(args)
        return "FunctionApproximator instance\n" + "".join("    {}: {}\n".format(key, value) for key, value in six.iteritems(self.parameters))

    def setup_from_json(self, json_file):
        """Set up the class before training from a json file"""
        if self.model is None:
            print("Setting up function approximator")
            self.parameters = utils.get_parameters_from_json(json_file)
            self.outdir = self.parameters["outdir"]
            self._setup_directories()
            shutil.copy(json_file, self.tmp_outdir)
            # setup network
            self.model = utils.network(self.n_extrinsic, self.n_intrinsic, self.parameters)
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
            self.model.compile(Adam(lr=self.parameters["learning_rate"], decay=self.parameters["lr_decay"]), loss=utils.KL, metrics=["mse"])
        else:
            print("Using " + self.parameters["loss"])
            self.model.compile(Adam(lr=self.parameters["learning_rate"], decay=self.parameters["lr_decay"]), loss=self.parameters["loss"], metrics=[utils.KL])

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
            #raise NotImplementedError("Accumulate not implemented yet")
            x = np.concatenate([self._accumulated_data[0], x], axis=0)
            y = np.concatenate([self._accumulated_data[1], y], axis=0)

        x, y = self._shuffle_data(x, y)
        n = len(y)
        x_train, x_val = np.array_split(x, [int(split * n)], axis=0)
        self.y_train, self.y_val = np.array_split(y, [int(split * n)], axis=0)
        if self.n_extrinsic and self.n_intrinsic:
            self.x_train = [x_train[:, :self.n_extrinsic], x_train[:, -self.n_intrinsic:]]
            self.x_val = [x_val[:, :self.n_extrinsic], x_val[:, -self.n_intrinsic:]]
        else:
            self.x_train = x_train
            self.x_val = x_val
        if accumulate:
            # save data before extrinsic/intrinsic split
            self._accumulated_data = (x_val, self.y_val)

        callbacks = []
        if self.parameters["patience"]:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=self.parameters["patience"]))
        # save best model during traing
        checkpoint = ModelCheckpoint(block_outdir + "model.h5", verbose=0, monitor="val_loss", save_best_only=True, mode="auto")
        callbacks.append(checkpoint)
        # more callbacks can be added by appending to callbacks
        history = self.model.fit(x=self.x_train, y=self.y_train, validation_data=(self.x_val, self.y_val), verbose=2, callbacks=callbacks, **self._training_parameters)
        training_y_pred = self.model.predict(self.x_train)
        self.model.load_weights(block_outdir + "model.h5")
        y_pred = self.model.predict(self.x_val).ravel()
        if plot:
            utils.make_plots(block_outdir, x=x_val, y_true=self.y_val, y_pred=y_pred, y_train_true=self.y_train, y_train_pred=training_y_pred, history=history, parameters=self.parameter_names)
        results_dict = {"x_train": self.x_train,
                        "x_val": self.x_val,
                        "y_train": self.y_train,
                        "y_val": self.y_val,
                        "training_preds": training_y_pred,
                        "y_pred": y_pred}
        results_dict.update(history.history)
        self.data_all["block{}".format(self._count)] = results_dict
        self._count += 1

    def _make_run_dir(self):
        """Check run count and make outdir"""
        run = 0
        while os.path.isdir(self.outdir + 'run{}'.format(run)):
            run += 1

        run_path = self.outdir + 'run{}/'.format(run)
        if not os.path.exists(run_path):
            os.mkdir(run_path)
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
