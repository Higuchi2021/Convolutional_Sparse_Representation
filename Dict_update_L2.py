import numpy as np
import Options
import Convert
import Util
import sys
from scipy.linalg import lu_factor, lu_solve

class Dict_update_L2():
    def __init__(self, opt, S):
        self.opt = opt
        self.S = S
        self.Sf = Convert.FFT_MN(self.S, self.opt)#S is (K,N)

        self.X = np.zeros((self.opt.K, self.opt.M, self.opt.N))
        self.G = np.zeros((self.opt.M, self.opt.N))
        self.H = np.zeros((self.opt.M, self.opt.N))

    def dict_update_L2(self, X):

        self.X = X
        self.Xf = Convert.FFT_KMN(self.X, self.opt)
        #print(self.D.shape)
        #print(self.Df.shape)
        
        for i in range(self.opt.dict_iteration):
            self.H_old = self.H.copy()

            self.D_update_LU()
            self.G_update()
            self.H_update()

        return self.G

    def D_update_LU(self):
        Xf = Util.create_block_stoructured_matrix(self.Xf)
        XfH = np.conj(Xf).T
        Sf = self.Sf.reshape(-1,).T
        Gf = Convert.FFT_MN(self.G, self.opt).reshape(-1,).T
        Hf = Convert.FFT_MN(self.H, self.opt).reshape(-1,).T

        #solve Ax=b by LU de
        Af = XfH @ Xf + self.opt.Rho*np.eye(self.opt.M*self.opt.N, self.opt.M*self.opt.N, dtype=complex)
        bf = XfH @ Sf + self.opt.Rho*(Gf - Hf)

        Df = np.linalg.solve(Af, bf)
        self.D = Convert.IFFT_MN(Df.reshape((self.opt.M, self.opt.N)), self.opt).real


    def D_update(self):
        Xf = Util.create_block_stoructured_matrix(self.Xf)
        XfH = np.conj(Xf).T
        Sf = self.Sf.reshape(-1,).T
        Gf = Convert.FFT_MN(self.G, self.opt).reshape(-1,).T
        Hf = Convert.FFT_MN(self.H, self.opt).reshape(-1,).T

        #print("D update processing...")
        a = (Gf - Hf)
        b = XfH
        c = np.linalg.inv(Xf@XfH + self.opt.Rho*np.eye(self.opt.K*self.opt.N, self.opt.K*self.opt.N, dtype=complex))
        d = Xf@(Gf - Hf) - Sf
        Df = a - b @ c @ d
        self.D = Convert.IFFT_MN(Df.reshape((self.opt.M, self.opt.N)), self.opt).real

    def G_update(self):
        #self.G = self.proxICPN(self.D + self.H)
        self.G = self.proxICPN_2Dlike(self.D + self.H)
    
    def H_update(self):
        self.H = self.H_old + self.D - self.G
    
    def proxICPN(self, d):
        PPT = np.zeros((self.opt.M, self.opt.N))
        PPT[:, 0:self.opt.L] = 1
        PPTd = PPT * d
        for i in range(self.opt.M):
            if np.linalg.norm(PPTd[i,:], ord=2) > 1.0:
                PPTd[i,:] /= np.linalg.norm(PPTd[i,:], ord=2)
        return PPTd

    def proxICPN_2Dlike(self, d):
        PPT = np.zeros((self.opt.M, self.opt.Image_Width, self.opt.Image_Width))
        PPT[:,:self.opt.Filter_Width, :self.opt.Filter_Width] = np.ones((self.opt.M, self.opt.Filter_Width, self.opt.Filter_Width))
        PPT = PPT.reshape((self.opt.M, self.opt.N))
        PPTd = PPT * d
        for i in range(self.opt.M):
            if np.linalg.norm(PPTd[i,:], ord=2) > 1.0:
                PPTd[i,:] /= np.linalg.norm(PPTd[i,:], ord=2)
        return PPTd


 