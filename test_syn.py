import os, time, pickle, random, glob
import numpy as np
from imageio import imread, imwrite
from skimage.measure import compare_psnr, compare_ssim

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import *
from dataloader import *
from Dataset.preprocess import *
from Dataset.postprocess import *


def evaluate_net(opt):

    src_path = opt["src_path"]
    test_items = opt["test_items"]
    dataset_name = opt["dataset_name"]
    result_path = opt["result_path"]
    iter_list = opt['iter_list']
    ckpt_dir = opt['ckpt_dir']
    NetName = opt['NetName']

    src_folder_list = []
    dst_path_list = []

    for item in test_items:
        tmp = sorted(glob.glob(src_path + item))
        src_folder_list.extend(tmp)
        dst_path_list.append(result_path + item)

    psnr = np.zeros((len(iter_list),len(src_folder_list)))
    ssim = np.zeros((len(iter_list),len(src_folder_list)))
    test_time = np.zeros((len(iter_list),len(src_folder_list)))
    for iter_num in range(len(iter_list)):

        if torch.cuda.is_available():
            model = torch.load(ckpt_dir + 'model_' + iter_list[iter_num] + '.pth')
            model = model.cuda()
        else:
            #continue
            model = torch.load(ckpt_dir + 'model_' + iter_list[iter_num] + '.pth', map_location='cpu')

        model.eval()

        #=================#
        for i in range(len(src_folder_list)):
            create_dir(dst_path_list[i])
            h5f = h5py.File(src_folder_list[i]+dataset_name, 'r')
            keys = list(h5f.keys())
            for ind in range(len(keys)):
                print(keys[ind])
                g = h5f[keys[ind]]
                mosaic_noisy = np.array(g['mosaic_noisy']).reshape(g['mosaic_noisy'].shape)
                mosaic_blur = np.array(g['mosaic_blur']).reshape(g['mosaic_blur'].shape)
                linRGB = np.array(g['linRGB']).reshape(g['linRGB'].shape)
                wb = np.array(g['wb']).reshape(g['wb'].shape)
                XYZ2Cam = np.array(g['XYZ2Cam']).reshape(g['XYZ2Cam'].shape)

                mosaic_noisy = mosaic_noisy[0, 0:(linRGB.shape[0]//16)*16, 0:(linRGB.shape[1]//16)*16, 0] # first one
                mosaic_blur = mosaic_blur[0, 0:(linRGB.shape[0]//16)*16, 0:(linRGB.shape[1]//16)*16, 0] # first one
                clean = linRGB[0:(linRGB.shape[0]//16)*16, 0:(linRGB.shape[1]//16)*16]

                mosaic_noisy = np.clip(mosaic_noisy, 0, 1)
                mosaic_blur = np.clip(mosaic_blur, 0, 1)
                clean = np.clip(clean, 0, 1)
                noisy = raw2rggb(mosaic_noisy)
                noisy= transforms.functional.to_tensor(noisy)
                noisy = noisy.unsqueeze_(0).float()
                blur = raw2rggb(mosaic_blur)
                blur= transforms.functional.to_tensor(blur)
                blur = blur.unsqueeze_(0).float()

                if torch.cuda.is_available():
                    noisy, blur = noisy.cuda(), blur.cuda()
                noisy, blur = Variable(noisy), Variable(blur)

                torch.cuda.synchronize()
                start_time = time.time()
                with torch.no_grad():
                    test_out = model(noisy, blur)
                    #test_out = model(torch.cat([noisy, blur], 1))
                    #test_out = model(noisy)
                torch.cuda.synchronize()
                if ind > 0:
                    test_time[iter_num][i] += (time.time() - start_time)

                # 计算loss
                rgb_out = test_out.cpu().detach().numpy().transpose((0,2,3,1))
                rgb = np.clip(rgb_out[0], 0, 1)

                rgb = postprocess(rgb, XYZ2Cam)
                imwrite(dst_path_list[i] + "%04d_out.png" % ind, np.uint8(rgb*255))

                clean = postprocess(clean, XYZ2Cam)

                #rgb, clean = np.round(rgb*255)/255, np.round(clean*255)/255
                psnr[iter_num][i] += compare_psnr(clean, rgb)

                if clean.ndim == 2:
                    ssim[iter_num][i] += compare_ssim(clean, rgb)
                elif clean.ndim == 3:
                    ssim[iter_num][i] += compare_ssim(clean, rgb, multichannel=True)

            test_time[iter_num][i] = test_time[iter_num][i] / ind
            psnr[iter_num][i] = psnr[iter_num][i] / (ind+1)
            ssim[iter_num][i] = ssim[iter_num][i] / (ind+1)

            h5f.close()

        #print psnr,ssim
    for iter_num in range(len(iter_list)):
        for i in range(len(src_folder_list)):
            #in_files = glob.glob(src_folder_list[i] + '*.png')
            print('iter_num: %8d, src_folder: %s: ' %(int(iter_list[iter_num]), src_folder_list[i]))
            print('psnr: %f, ssim: %f, average time: %f' % (psnr[iter_num][i], ssim[iter_num][i], test_time[iter_num][i]))
            #print('psnr: %f' % (psnr[iter_num][i] / len(in_files)))

    return 0


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    opt = {
            "src_path": "/ssd2T_3/",
            "test_items": ["multiexposure/"],
            "dataset_name": "test_2.h5",

            "result_path": "/hdd4T_1/cm/codes/multiexposure/result_png/LSFNet_L1/",
            "ckpt_dir": "/hdd4T_1/cm/codes/multiexposure/ckpt/LSFNet_L1/",

            "iter_list": ['0300'],
            "NetName": LSFNet,

    }


    evaluate_net(opt)
