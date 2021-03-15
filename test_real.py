import os, time, pickle, random, glob
import numpy as np
from imageio import imread, imwrite

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import *
from Dataset.preprocess import *
from Dataset.postprocess import *


def crop_patch(img, patch_size=(150, 150), stride=150):

    img_size = img.shape
    count = 0
    img_list = []

    pos = [(x, y) for x in range(patch_size[1], img_size[1] - patch_size[1], stride) for y in
           range(patch_size[0], img_size[0] - patch_size[0], stride)]

    for (xt, yt) in pos:
        cropped_img = img[yt - patch_size[0]:yt + patch_size[0], xt - patch_size[1]:xt + patch_size[1]]

        img_list.append(cropped_img)

    return img_list

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
                mosaic_noisy_2 = np.array(g['mosaic_noisy_2']).reshape(g['mosaic_noisy_2'].shape)
                wb = np.array(g['wb']).reshape(g['wb'].shape)
                XYZ2Cam = np.array(g['XYZ2Cam']).reshape(g['XYZ2Cam'].shape)

                #mosaic_noisy = mosaic_noisy[0:(mosaic_noisy.shape[0]//32)*32, 0:(mosaic_noisy.shape[1]//32)*32]
                #mosaic_blur = mosaic_blur[0:(mosaic_blur.shape[0]//32)*32, 0:(mosaic_blur.shape[1]//32)*32]

                #ratio = 30
                ratio = 10
                noisy = raw2rggb(mosaic_noisy) * ratio
                noisy = np.clip(noisy, 0, 1)
                blur = raw2rggb(mosaic_blur/3)

                img_list = crop_patch(np.concatenate([noisy, blur], 2), (256, 256), 480)

                for num in range(len(img_list)):

                    patch = transforms.functional.to_tensor(img_list[num])
                    patch = patch.unsqueeze_(0).float()

                    if torch.cuda.is_available():
                        patch = patch.cuda()
                    patch = Variable(patch)

                    test_out = model(patch[:,0:4,:,:], patch[:,4:8,:,:])

                    rgb_out = test_out.cpu().detach().numpy().transpose((0,2,3,1))
                    rgb = np.clip(rgb_out[0], 0, 1)

                    rgb = postprocess(rgb, XYZ2Cam, hdr_compress=True)
                    imwrite(dst_path_list[i] + keys[ind]+"_%04d_out.png" % (num), np.uint8(rgb*255))

                '''
                noisy= transforms.functional.to_tensor(noisy)
                noisy = noisy.unsqueeze_(0).float()

                blur= transforms.functional.to_tensor(blur)
                blur = blur.unsqueeze_(0).float()

                if torch.cuda.is_available():
                    noisy, blur = noisy.cuda(), blur.cuda()
                noisy, blur = Variable(noisy), Variable(blur)

                torch.cuda.synchronize()
                start_time = time.time()
                with torch.no_grad():
                    test_out = model(noisy, blur)
                torch.cuda.synchronize()
                if ind > 0:
                    test_time[iter_num][i] += (time.time() - start_time)

                #
                rgb_out = test_out.cpu().detach().numpy().transpose((0,2,3,1))
                rgb = np.clip(rgb_out[0], 0, 1)
                rgb = postprocess(rgb, XYZ2Cam, hdr_compress=True)
                imwrite(dst_path_list[i] + "%04d_out.png" % ind, np.uint8(rgb*255))
                '''

            h5f.close()

        #print psnr,ssim
    for iter_num in range(len(iter_list)):
        for i in range(len(src_folder_list)):
            #in_files = glob.glob(src_folder_list[i] + '*.png')
            print('iter_num: %8d, src_folder: %s: ' %(int(iter_list[iter_num]), src_folder_list[i]))
            print('average time: %f' % (test_time[iter_num][i]))

    return 0



if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    opt = {
            "src_path": "./Dataset/",
            "test_items": ["test/"],
            "dataset_name": "test_real.h5",

            "result_path": "./test_real_sample/",
            'ckpt_dir': "./ckpt/LSFNet_L1_hdr/",

            "iter_list": ['0300'],
            "NetName": LSFNet,
    }


    evaluate_net(opt)
