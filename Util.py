import numpy as np
#/////////////////////////////////////////////////////////
#二次元配列辞書の一次元配列化と辞書のパディング
#/////////////////////////////////////////////////////////
def dict_vectorize(D, opt):
    D_pad = np.zeros((opt.M, opt.N))
    for i in range(opt.M):
        D_pad[i][0:opt.Filter_Width*opt.Filter_Width] = D[i].reshape(-1,)
    return D_pad

#/////////////////////////////////////////////////////////
#another padding method
#/////////////////////////////////////////////////////////
def dict_vectorize_2Dlike(D, opt):
    D_pad = np.zeros((opt.M, opt.Image_Width, opt.Image_Width))
    D_pad[:,:opt.Filter_Width,:opt.Filter_Width] = D
    return D_pad.reshape((opt.M, opt.N))

#/////////////////////////////////////////////////////////
#画像の正規化
#/////////////////////////////////////////////////////////
def normalize(img):
    mean = img.mean()
    std = np.std(img)
    return (img-mean) / std

#/////////////////////////////////////////////////////////
#a vector to a diagonalized matrix
#/////////////////////////////////////////////////////////
def create_diagmatrix(x):
    N = len(x)
    X = np.zeros((N, N), dtype=np.complex)
    for i in range(N):
        X[i,i] = x[i]
    return X

#arrange diagonalized matrixes to block structured matrixes 
def create_block_stoructured_matrix(x):
    #shape of X is (K, M, N)
    #then this returns (KN, MN)
    K = x.shape[0]
    M = x.shape[1]
    N = x.shape[2]
    #print("block K M N")
    #print(K)
    #print(M)
    #print(N)
    X = np.zeros((K*N, M*N), dtype=np.complex)
    for i in range(K):
        for j in range(M):
            X[i*N:(i+1)*N, j*N:(j+1)*N] = create_diagmatrix(x[i,j,:])
    return X
