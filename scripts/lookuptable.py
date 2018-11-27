#!/usr/bin/env python

import sys
import numpy as np
from scipy.integrate import quad
from scipy.spatial import cKDTree

import csv

def get_integral(r0,r1):
    """ function to call integral fucntion"""
    def calculate_integral(a,b):
        """ calculate the integral over a given range of r"""
        f = lambda r :  r**2. * np.exp(-a/2.*r**2. + b/r)
        # numerical integration
        y, abserr = quad(f, r0, r1 )
        return y, abserr

    return calculate_integral

def make_LUT():
    """
    Makes the lookup table (with hard coded ranges and number of entries)
    """
    # define distance range
    r0 = 1
    r1 = 101
    # get integral to calculate for different values of A and B
    cal_int = get_integral(r0,r1)
    # number of points and values for A and B
    N = 500
    A = np.linspace(0,100, N)
    B = np.linspace(0,100, N)
    # create .cvs file
    with open('LUT.csv', mode='w') as LUT_file:
        # use , as delimeter
        LUT = csv.writer(LUT_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # Can add columns
        #LUT.writerow(["A", "B", "L", "AE"])

        # look over values and calculate integral
        counter = 0
        for a in A:
            for b in B:
                # integral
                y,ae = cal_int(a,b)
                # write to file
                LUT.writerow([a,b,y,ae])
                counter +=1
                # keep track of progress
                if counter % 1000 ==0:
                    print(f'Progress: {counter} / {N**2}',end='\r', flush=True)
        print('')

def load_LUT(return_LUT=False):
    """
    Load lookup table and return a cDKTree and the value for each point
    """
    # load file
    LUT = np.genfromtxt('LUT.csv', delimiter=',')
    # return LUT if requested
    if return_LUT:
        return LUT
    # spatial positions
    pos = LUT[:, :-2]    # last two dims are integral and error
    val = LUT[:, -2]
    # delete LUT to save memory
    del LUT
    # make tree
    tree = cKDTree(pos)
    print('Loaded lookup table...')
    return tree, val

def get_values(coords=((0., 0.), (1., 1.))):
    """
    Get the values from the LUT for a given set of points
    """
    # get tree
    tree, val = load_LUT()
    # choose values to interpolate
    dist, ind = tree.query(coords, k=2)
    # get vectors of closest entries
    d1, d2 = dist.T
    # and their values
    v1, v2 = val[ind].T
    # estimate value
    V = (d1)/(d1 + d2)*(v2 - v1) + v1

    print('Requested points | Interpolated values')
    for c,v in zip(coords, V):
        print(f'     {c} | {v}')

    return V

def main():
    """
    Run different operations on a look up table
    """
    if len(sys.argv) > 1:
        task = sys.argv[1]
        if task =='create':
            print('Creating lookup table...')
            make_LUT()
            exit(0)

        elif task == 'load':
            print('Loading lookup table...')
            load_LUT()
            exit(0)

        else:
            print('Unkown task...')
            exit(1)
    else:
        print('No task specified...')
        exit(0)

if __name__ == "__main__":

    main()

