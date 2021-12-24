import numpy as np
import Options
import Convert
import Util
import sys


class Coef_update_L2():
    def __init__(self, opt, S):
        self.opt = opt
        self.S = S
        self.Sf = Convert.FFT_MN(self.S, self.opt)#S is (K,N)

        self.X = np.zeros((self.opt.K, self.opt.M, self.opt.N))
        self.Y = np.zeros((self.opt.K, self.opt.M, self.opt.N))
        self.U = np.zeros((self.opt.K, self.opt.M, self.opt.N))

    def coef_update_L2(self, D):

        self.D = D
        self.Df = Convert.FFT_MN(self.D, self.opt)
        #print(self.D.shape)
        #print(self.Df.shape)
        
        for i in range(self.opt.coef_iteration):
            self.U_old = self.U.copy()
            #print("X_update")
            self.X_update()
            #print("Y_update")
            self.Y_update()
            #print("U_update")
            self.U_update()
        return self.Y

    def X_update(self):
        DfH = np.conj(self.Df)
        #print("shape")
        #print(self.Y.shape)
        #print(self.Y[k].shape)
        Yf = Convert.FFT_KMN(self.Y, self.opt)
        Uf = Convert.FFT_KMN(self.U, self.opt)

        #print("X update processing...")
        a = DfH / (self.opt.Rho * np.ones((self.opt.N), dtype=complex) + np.sum(self.Df*DfH, axis=0))
        b = np.sum(self.Df*(Yf-Uf), axis=1) - self.Sf
        c = Yf - Uf
        #broadcastのためのreshape
        a = np.tile(a, (self.opt.K,1,1))
        b = b.reshape(self.opt.K,1,self.opt.N)

        Xf = c - (a * b)

        self.X = Convert.IFFT_KMN(Xf, self.opt).real
        #print("self.X")
        #print(self.X[0,0,0])

    def Y_update(self):
        x = self.X + self.U
        alpha = self.opt.Lambda / self.opt.Rho
        #print("y_update")
        #print(x.shape)
        self.Y = self.prox_L1(x, alpha)
        #print(self.Y.shape)
        #print("self.Y")
        #print(self.Y[0,0,0])

    def U_update(self):
        #print("U_old")
        #print(self.U_old[0,0,0])
        self.U = self.U_old + self.X - self.Y
        #print("UXY")
        #print(self.U[0,0,0])
        #print(self.X[0,0,0])
        #print(self.Y[0,0,0])
    #/////////////////////////////////////////////////////////
    #L1のprox(ソフト閾値関数)
    #/////////////////////////////////////////////////////////
    def prox_L1(self, x, alpha):
        #print("prox_L1")
        #print(x[0,0,0])
        res = np.sign(x) * (np.clip(np.abs(x) - alpha, 0, float('Inf')))
        #print(res[0,0,0])
        return res