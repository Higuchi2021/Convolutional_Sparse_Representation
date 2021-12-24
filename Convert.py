import numpy as np
import Options

#///////////////////////////////////////////////////////////
#FFT to each vector of MN matrix
#///////////////////////////////////////////////////////////
def FFT_MN(D, opt):
    Df = np.fft.fft(D, axis=1)
    return Df 
def IFFT_MN(Df, opt):
    D = np.fft.ifft(Df, axis=1)
    return D

def FFT_KMN(D, opt):
    Df = np.fft.fft(D, axis=2)
    return Df 
def IFFT_KMN(Df, opt):
    D = np.fft.ifft(Df, axis=2)
    return D


