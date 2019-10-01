
from __future__ import print_function

from gwfa.utils import set_keras_device
# set up the device to use
# by default this is the gpu
# but can be set to cpu as well
set_keras_device()

from gwfa import function_approximator
from gwfa import data
