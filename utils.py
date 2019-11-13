import os

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def onlyfiles(directory):
    return [f
            for f
            in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))]

import numpy as np
from numpy import pi
import pylab as plt
from colorsys import hls_to_rgb

def colorize(x):
    print(x.shape)
    z = x[:,:,0] + 1j * x[:,:,1]
    r = np.abs(z)
    arg = np.angle(z) 

    h = (arg + pi)  / (2 * pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2)
    c = c.swapaxes(0,1)
    return c
