import numpy as np
class DictLearn_Option():
    def __init__(self, K, M, N, L, Image_Width, Filter_Width, Lambda, Rho, coef_iteration, dict_iteration):
        """
        K:　画像の枚数
        M:  辞書の枚数
        N:  画像の画素数
        L:　辞書の画素数
        Image_Width:　画像の一辺の画素数
        Filter_Width:　辞書の一辺の画素数
        Lambda: 非ゼロ係数
        Rho:    ADMM内での係数
        coef_iteration: 係数更新の繰り返し回数
        dict_iteration: 辞書更新の繰り返し回数
        """
        self.K = K
        self.M = M
        self.N = N
        self.L = L
        self.Image_Width = Image_Width 
        self.Filter_Width = Filter_Width
        self.Lambda = Lambda
        self.Rho = Rho
        self.coef_iteration = coef_iteration
        self.dict_iteration = dict_iteration

class Decode_Fourier_Option():
    def __init__(self, N, M, L, Myu, Rho, iteration, filter):
        self.N = N
        self.M = M
        self.L = L            #圧縮後の次元
        self.Myu = Myu
        self.Rho = Rho
        self.iteration = iteration
        self.nonzero_index = np.zeros(self.L, dtype=np.int)   

        if filter == "low":           #高周波成分を圧縮  
            index = np.arange(self.N, dtype=np.int)
            self.nonzero_index[:L//2] = index[:L//2]
            self.nonzero_index[L//2:] = index[self.N-L//2:]
            print(self.nonzero_index[:L//2])
            print(self.nonzero_index[L//2:])
        
        elif filter == "random":      #ランダムに圧縮
            index = np.arange(self.N//2-1, dtype=np.int) + 1
            a = np.zeros(self.L//2)
            a[1:L//2] = np.random.permutation(index)[:L//2-1]
            b = self.N - a
            self.nonzero_index[:L//2] = a
            self.nonzero_index[L//2:] = b
            self.nonzero_index[self.L//2] = self.N//2  
        
class Decode_Random_Option():
    def __init__(self, N, M, R, Myu, Rho, iteration):
        """
        N:  画像の画素数
        M:  辞書の枚数
        R:　圧縮行列の各列ベクトルの次元
        Myu:　圧縮係数
        Rho:　ADMM係数
        iteration:係数の更新回数
        """
        self.N = N
        self.M = M
        self.R = R
        self.Myu = Myu
        self.Rho = Rho
        self.iteration = iteration
        if(R == self.N):
            self.Phi = np.eye(self.L)
        else:
            self.Phi = np.random.normal(0, 1/self.N, (self.R, self.N))
        #self.Phi = np.random.normal(0, 1, (self.L, self.N))
        