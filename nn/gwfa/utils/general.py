
import shutil
import os

def set_keras_device(device="GPU0", gpu_fraction=0.3):
    # get type of device: CPU or GPU
    device_type = device.rstrip("0123456789")
    import tensorflow as tf
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import keras.backend as K
    if device_type == "GPU" or device_type == "gpu":
        # if no gpu is specified will use 0
        if device_type is not device:
            device_number = device[len(device_type)]
        else:
            device_number = "0"
        # set available gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = device_number
        print("Setting up Keras to use GPU with miminal memory on {}".format(device_number))
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
    elif device_type == "CPU" or device_type == "cpu":
        print("Setting up Keras to use CPU")
        config = tf.compat.v1.ConfigProto(device_count = {"GPU": 0})
    # set up session
    K.tensorflow_backend.set_session(tf.compat.v1.Session(config=config))
    sess = tf.compat.v1.Session(config=config)

def fuzz():
    """Fuzz factor to avoid NaNs"""
    return 1e-6

def copytree(src, dst, symlinks=False, ignore=None):
    """Move the contents of a directory to a specified directory"""
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def make_run_dir(outdir):
    """Check run count and make outdir"""
    run = 0
    while os.path.isdir(outdir + 'run{}'.format(run)):
        run += 1

    run_path = outdir + 'run{}/'.format(run)
    if not os.path.exists(run_path):
        os.mkdir(run_path)
    return run_path
