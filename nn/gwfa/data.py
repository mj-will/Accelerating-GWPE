import os
import numpy as np
import pandas as pd

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

    def prep_data_chain(self, block_size=1000, norm_logL=False, norm_intrinsic=False, norm_extrinsic=False):
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
            self.logL = np.apply_along_axis(self.normalize_logL, 0, self.logL)[:-diff].reshape(self.N_blocks, self.block_size)
        else:
            self.logL = self.logL[:-diff].reshape(self.N_blocks, self.block_size)
        # intrinsic
        if self.N_intrinsic:
            if norm_intrinsic:
                self.intrinsic_parameters = np.apply_along_axis(self.normalize_parameters, 0, self.intrinsic_parameters)[:-diff, :].reshape(self.N_blocks, self.block_size, self.N_intrinsic)
            else:
                self.intrinsic_parameters = self.intrinsic_parameters[:-diff, :].reshape(self.N_blocks, self.block_size, self.N_intrinsic)
        # extrinsic
        if self.N_extrinsic:
            if norm_extrinsic:
                self.extrinsic_parameters = np.apply_along_axis(self.normalize_parameters, 0, self.extrinsic_parameters)[:-diff, :].reshape(self.N_blocks, self.block_size, self.N_extrinsic)
            else:
                self.extrinsic_parameters = self.extrinsic_parameters[:-diff, :].reshape(self.N_blocks, self.block_size, self.N_extrinsic)
        print('X shape:', self.intrinsic_parameters.shape, self.extrinsic_parameters.shape)
        print('Y shape:', self.logL.shape)
