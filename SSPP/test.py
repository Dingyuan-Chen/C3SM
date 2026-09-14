from tqdm import tqdm
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.CRITICAL)

from utils import *
from ReFusion import ReFusion,LPN
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def spectral_shift_loss(fuse, opt, eps=1e-6, lambda_freq=0.2):
    def normalize_intensity(x):
        mean = x.mean(dim=[2,3], keepdim=True)
        std  = x.std(dim=[2,3], keepdim=True)
        return (x - mean) / (std + eps)

    opt_n = normalize_intensity(opt)
    fuse_n = normalize_intensity(fuse)

    L_spatial = F.mse_loss(opt_n, fuse_n)

    def power_spectrum(x):
        fft = torch.fft.fft2(x)
        mag = torch.abs(fft)

        mag = torch.pow(mag + eps, 0.5)
        mag = mag / (mag.sum(dim=[2,3], keepdim=True) + eps)
        return mag

    P_opt = power_spectrum(opt)
    P_fuse = power_spectrum(fuse)

    L_freq = F.mse_loss(P_opt, P_fuse)

    L = lambda_freq * L_spatial + L_freq

    return L

for task in ["IVIF"]:
    print("test task: "+task)
    if task == "IVIF":
        path_model=r"./exp/spectral_TIR/model/ckpt_50.pth"
        path_img1=r"/{path_to_files}/train/TIR/images_ir"
        path_img2=r"/{path_to_files}/train/TIR/images"
        path_result=r"/{path_to_files}/train/TIR/C3SM_spec"

    Fusion_model= ReFusion().to(device)
    Fusion_model.load_state_dict(torch.load(path_model))

    Loss_model = LPN().to(device)
    Loss_model.load_state_dict(torch.load(path_model.replace('ckpt', 'lpn_ckpt')))

    with open("/{path_to_files}/train/TIR/TIR_ssi.txt", "w") as f:

        with torch.no_grad():
            for imgname in tqdm(os.listdir(path_img1)):
                img1=image_read(os.path.join(path_img1, imgname))
                img2=image_read(os.path.join(path_img2, imgname))

                img1_YCrCb=cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)
                img2_YCrCb=cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)

                if is_grayscale(img1):
                    if is_grayscale(img2):
                        CrCb=None
                    else:
                        CrCb=img2_YCrCb[:,:,1:]
                else:
                    if is_grayscale(img2):
                        CrCb=img1_YCrCb[:,:,1:]
                    else:
                        CrCb=fuse_CrCb(img1_YCrCb[:,:,1:],img2_YCrCb[:,:,1:])

                img1=img1_YCrCb[:,:,0][np.newaxis,np.newaxis,...]/255
                img2=img2_YCrCb[:,:,0][np.newaxis,np.newaxis,...]/255
                img1 = ((torch.FloatTensor(img1))).to(device)
                img2 = ((torch.FloatTensor(img2))).to(device)

                data_Fuse=Fusion_model(torch.cat((img1,img2),1))
                data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
                fused_image = np.squeeze((data_Fuse * 255).cpu().numpy())

                w1, w2 = Loss_model(torch.cat((img1, img2), 1))
                w1, w2 = w1.detach(), w2.detach()

                W = torch.sqrt(w1[:,0:1,:,:].pow(2) + w2[:,0:1,:,:].pow(2))  # [1,1,H,W]
                W_img = W.mean()
                W_img = torch.maximum(W_img, torch.tensor(0.1, device=W_img.device))
                L_s = spectral_shift_loss(data_Fuse, img1)
                ssi = torch.sigmoid(W_img * L_s)

                print(imgname, ssi)
                f.write(f"{imgname} {ssi.item()}\n")
                image_save(fused_image, imgname.split(sep='.')[0], path_result, CrCb)