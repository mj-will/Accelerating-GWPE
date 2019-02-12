
import shutil
import os

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
