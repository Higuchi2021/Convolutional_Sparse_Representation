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
            for j in range(self.opt.K):
                self.U_old = self.U.copy()
                self.X_update(j)
            self.Y_update()
            self.U_update()
        return self.Y

    def X_update(self, k):
        Df = Util.create_block_stoructured_matrix(np.array([self.Df]))
        DfH = np.conj(Df).T
        Skf = self.Sf[k,:].reshape(-1).T
        #print("shape")
        #print(self.Y.shape)
        #print(self.Y[k].shape)
        Ykf = Convert.FFT_MN(self.Y[k,:,:], self.opt).reshape(-1,).T
        Ukf = Convert.FFT_MN(self.U[k,:,:], self.opt).reshape(-1,).T

        #print("X update processing...")
        a = DfH @ np.linalg.inv(self.opt.Rho*np.eye(self.opt.N) + Df@DfH)
        b = (Skf + Df@(Ykf - Ukf))
        c = (Ykf - Ukf)
        Xkf = a @ b - c
        #Xkf = DfH @ (self.opt.Rho*np.eye(self.opt.N) + Df@DfH)**-1 @ (Skf + DfH@(Ykf - Ukf)) - (Ykf - Ukf)
        self.X[k,:,:] = Convert.IFFT_MN(Xkf.reshape((self.opt.M, self.opt.N)), self.opt).real

    def Y_update(self):
        x = self.X + self.U
        alpha = self.opt.Lambda / self.opt.Rho
        #print("y_update")
        #print(x.shape)
        self.Y = self.prox_L1(x, alpha)
        #print(self.Y.shape)

    def U_update(self):
        self.U = self.U_old + self.X - self.Y
        
    #/////////////////////////////////////////////////////////
    #L1のprox(ソフト閾値関数)
    #/////////////////////////////////////////////////////////
    def prox_L1(self, x, alpha):
        return np.sign(x) * (np.clip(np.abs(x) - alpha, 0, float('Inf')))