import numpy as np
import Convert
from sporco.metric import psnr
from skimage.metrics import structural_similarity
from sporco.util import tiledict
from sporco.plot import imview

def get_tiled_D(D, opt, *, titlename='Dictionary'):
    D_copy = D[:,:opt.L].reshape((opt.M, opt.Filter_Width, opt.Filter_Width))
    D_copy = np.transpose(D_copy,(1, 2, 0))
    Output = D_copy
    #Output = np.asnumpy(D_copy)
    imview(tiledict(Output), fgsz=(7,7), title=titlename)

def get_tiled_D_2Dlike(D, opt, *, titlename='Dictionary_2Dlike_vectrize'):
    D_copy = D.reshape((opt.M, opt.Image_Width, opt.Image_Width))[:,:opt.Filter_Width,:opt.Filter_Width]
    D_copy = np.transpose(D_copy, (1, 2, 0))
    Output = D_copy
    #Output = np.asnumpy(D_copy)
    imview(tiledict(Output), fgsz=(7,7), title=titlename)

def get_tiled_Xk(Xk, opt, *, titlename='Coefficients'):
    Xk_copy = Xk.reshape((opt.M, opt.Image_Width, opt.Image_Width))
    Xk_copy = np.transpose(Xk_copy,(1, 2, 0))
    Output = Xk_copy
    #Output = np.asnumpy(Xk_copy)
    imview(tiledict(Output), fgsz=(7,7), title=titlename)

def get_tiled_Img(Img, opt, *, titlename='Images'):
    Img_copy = Img.reshape((opt.K, opt.Image_Width, opt.Image_Width))
    Img_copy = np.transpose(Img_copy,(1, 2, 0))
    Output = Img_copy
    #Output = np.asnumpy(Img_copy)
    imview(tiledict(Output), fgsz=(7,7), title=titlename)

def get_reconstructed_img(D, X, opt, *, show=True, titlename='Reconstructed Images'):
    Df = Convert.FFT_MN(D, opt).reshape(1,*D.shape)
    Xf = Convert.FFT_KMN(X, opt)
    DXf = np.sum(Df * Xf, axis=1)
    print(DXf.shape)
    DX = Convert.IFFT_MN(DXf, opt).real
    if show:
        get_tiled_Img(DX, opt, titlename=titlename)
    return DX

def get_L0_norm(X, opt):
    for i in range(opt.K):
        print("{}th L0 norm: ".format(i), np.sum(np.abs(X[i,:,:]) != 0))

def get_psnr_and_ssim(S, S_DX, opt):
    imgs = S.reshape((opt.K, opt.Image_Width, opt.Image_Width))
    reconsts = S_DX.reshape((opt.K, opt.Image_Width, opt.Image_Width))
    Output_imgs = imgs
    Output_reconsts = reconsts
    #Output_imgs = np.asnumpy(imgs)
    #Output_reconsts = np.asnumpy(reconsts)
    for i in range(opt.K):
        print("{}th PSNR: ".format(i), psnr(Output_imgs[i], Output_reconsts[i]))
        print("{}th SSIM: ".format(i), structural_similarity(Output_imgs[i], Output_reconsts[i], data_range=Output_imgs[i].max()-Output_imgs[i].min()))
