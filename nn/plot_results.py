
import sys
from utils import make_plots_multiple

def main():
    print(len(sys.argv))
    inputdir = sys.argv[1]
    outputdir = sys.argv[2]
    make_plots_multiple(inputdir, outputdir, blocks = "all")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError("Missing output dir")
    else:
        main()

