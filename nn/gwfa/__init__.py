
from gwfa import function_approximator
from gwfa import data

import tensorflow as tf
import keras.backend as K

# TensorFlow wizardry
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))
