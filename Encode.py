import numpy as np
import Options
import Convert
import Util

def encode_fourier(S, opt):
    Sf = Convert.FFT_MN(S, opt)
    Y = Sf[opt.nonzero_index]
    #print("encoding fourier...")
    #print(Sf.shape)
    #print(Y.shape)
    return Y

def encode_random(S, opt):
    Y = opt.Phi @ S
    #print("encoding random...")
    #print(opt.Phi.shape)
    #print(S.shape)
    return Y