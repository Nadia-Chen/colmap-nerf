import torch
from kornia.losses import ssim as dssim
import torch
import os
import numpy as np
from tqdm import tqdm
import imageio
from argparse import ArgumentParser


import metrics

from datasets import dataset_dict
from datasets.depth_utils import *

from PIL import Image
from torchvision import transforms as T

import lpips
lpips_alex = lpips.LPIPS(net='alex') # best forward scores
imgs, psnrs, ssims, lpips_alexs, lpips_vggs, maes, mses = [], [], [], [], [], [], []
dir_name = os.path.join(args.save_dir, f'results/{args.dataset_name}/{args.scene_name}')
os.makedirs(dir_name, exist_ok=True)

toTensor = T.ToTensor()
normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

f_list = os.listdir(dir_name)
idx_list = [i[0:3] for i in f_list]
for i in tqdm(range(len(dataset))):
    sample = dataset[i]
    image_pre_path = os.path.join(dir_name, f_list[idx_list.index(f'{i:03d}')])
    img_pred = Image.open(image_pre_path).convert('RGB')
    img_pred = toTensor(img_pred) # (3, h, w)
    if args.dataset_name == 'blender':
        w, h = args.img_wh
    else:
        w, h = sample['img_wh']

    normalize_img_pre = normalize(img_pred).unsqueeze(0)
        
    img_pred_ = (img_pred.permute(1, 2, 0).numpy()*255).astype(np.uint8)
    imgs += [img_pred_]

    rgbs = sample['rgbs']
    img_gt = rgbs.view(h, w, 3)
    if args.dataset_name == 'phototourism':
        psnrs += [metrics.psnr(img_gt[:,w//2:,:], img_pred.permute(1, 2, 0)[:,w//2:,:]).item()]
        ssims += [metrics.ssim(img_gt[:,w//2:,:].permute(2, 0, 1)[None,...], img_pred[:, :, w//2:][None,...]).item()]
        lpips_alexs += [lpips_alex((img_gt[:,w//2:,:].permute(2, 0, 1)[None,...]*2-1), normalize_img_pre[...,w//2:]).item()]
        mses += [((img_gt[:,w//2:,:] - img_pred.permute(1, 2, 0)[:,w//2:,:])**2).mean().item()]
    else:
        psnrs += [metrics.psnr(img_gt, img_pred.permute(1, 2, 0)).item()]
        ssims += [metrics.ssim(img_gt.permute(2, 0, 1)[None,...], img_pred[None,...]).item()]
        lpips_alexs += [lpips_alex((img_gt.permute(2, 0, 1)[None,...]*2-1), normalize_img_pre).item()]
        mses += [((img_gt - img_pred.permute(1, 2, 0))**2).mean().item()]

if args.dataset_name == 'blender' or \
      (args.dataset_name == 'phototourism' and args.split == 'test'):
        imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}_30.{args.video_format}'),
                        imgs, fps=30)
    
mean_psnr = np.mean(psnrs)
mean_ssim = np.mean(ssims)
mean_lpips_alex = np.mean(lpips_alexs)
mean_mse = np.mean(mses)
with open(os.path.join(dir_name, 'result.txt'), "a") as f:
        f.write(f'metrics : \n')
        f.write(f'Mean PSNR : {mean_psnr:.4f}\n')
        f.write(f'Mean SSIM : {mean_ssim:.4f}\n')
        f.write(f'Mean LIPIS_alex : {mean_lpips_alex:.4f}\n')
        f.write(f'Mean MSE : {mean_mse:.4f}\n')
print('Done')


def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def ssim(image_pred, image_gt, reduction='mean'):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    dssim_ = dssim(image_pred, image_gt, 3, reduction) # dissimilarity in [0, 1]
    return 1-2*dssim_ # in [-1, 1]