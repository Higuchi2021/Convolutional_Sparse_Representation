import numpy as np
from PIL import Image
import Util
import matplotlib.pyplot as plt

def Image_In(K, is_test = False):
    if(is_test):
        ims = np.zeros((K, 128 * 128))
        for i in range(K):
            ims[i,:] = Util.normalize(np.array(Image.open("data/test/data0"+str(i)+".png"), dtype=np.float).flatten())
        return ims
    else :
        ims = np.zeros((K, 128 * 128))
        for i in range(K):
            ims[i,:] = Util.normalize(np.array(Image.open("data/train/data0"+str(i)+".png"), dtype=np.float).flatten())
        return ims



