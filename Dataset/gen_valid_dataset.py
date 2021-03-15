import h5py
import matplotlib.image as mpimg
import os, random
import numpy as np
import glob
import math
import scipy.io as sio
import rawpy
from Dataset.preprocess import *


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process(mosaic_blur, linRGB, wb, burst_num):
    gain = np.random.uniform(1.3, 2.3)
    mosaic_blur = mosaic_blur * gain
    linRGB = linRGB * gain

    ratio = 30
    thr = gain
    mask = (linRGB[:,:,0]<thr) | (linRGB[:,:,1]<thr) | (linRGB[:,:,-1]<thr)
    mask = mask.astype(np.float)
    mask = np.tile(mask[:,:,np.newaxis], [1,1,3])
    linRGB_short = (linRGB / ratio) * mask + linRGB * (1 - mask)

    # dark current
    #dark_noise = np.random.uniform(0.01, 0.03) / ratio
    #mosaic_noisy = mosaic_noisy - dark_noise
    #color_distortion_r = np.random.uniform(0.7, 0.9)
    #color_distortion_b = np.random.uniform(0.7, 0.9)
    color_distortion_r = np.random.uniform(0.8, 0.95)
    color_distortion_b = np.random.uniform(0.8, 0.95)
    linRGB_short[:,:,0] = linRGB_short[:,:,0] * color_distortion_r
    linRGB_short[:,:,2] = linRGB_short[:,:,2] * color_distortion_b
    # nosie
    mosaic_noisy = mosaic_bayer(linRGB_short)
    sigma_s = np.random.uniform(0.00001, 0.001) * 0.1
    sigma_r = np.random.uniform(0.01, 0.1) * sigma_s
    #meta['sigma'] = np.stack([sigma_s, sigma_r],-1)
    sigma_total = np.sqrt(sigma_s * mosaic_noisy + sigma_r)
    mosaic_noisy_list = []
    for num in range(burst_num):
        mosaic_noisy_list.append(mosaic_noisy + np.random.normal(scale=sigma_total, size=mosaic_noisy.shape))

    for num in range(burst_num):
        sigma_total = np.sqrt(sigma_s * mosaic_blur[num] + sigma_r)
        mosaic_blur[num] = mosaic_blur[num] + np.random.normal(scale=sigma_total, size=mosaic_blur[num].shape)

    mosaic_blur = np.clip(mosaic_blur, 0.0, 1.0)
    mosaic_noisy = np.clip(np.stack(mosaic_noisy_list, 0), 0.0, 1.0)
    # quantization
    mosaic_noisy = np.round(mosaic_noisy * (16383 - 512))  / (16383 - 512)
    mosaic_noisy = mosaic_noisy * ratio

    gr, gb = wb[0], wb[2]

    # Mosaic and wb
    raw_R = mosaic_blur[:, 0::2, 0::2] * gr
    raw_Gr = mosaic_blur[:, 0::2, 1::2]
    raw_Gb = mosaic_blur[:, 1::2, 0::2]
    raw_B = mosaic_blur[:, 1::2, 1::2] * gb
    mosaic_blur = np.stack([raw_R, raw_Gr, raw_Gb, raw_B], -1)

    raw_R = mosaic_noisy[:, 0::2, 0::2] * gr
    raw_Gr = mosaic_noisy[:, 0::2, 1::2]
    raw_Gb = mosaic_noisy[:, 1::2, 0::2]
    raw_B = mosaic_noisy[:, 1::2, 1::2] * gb
    mosaic_noisy = np.stack([raw_R, raw_Gr, raw_Gb, raw_B], -1)

    linear_R =  linRGB[:, :, 0] * gr
    linear_G =  linRGB[:, :, 1]
    linear_B =  linRGB[:, :, 2] * gb
    linRGB = np.stack([linear_R, linear_G, linear_B], -1)

    #mosaic_noisy = np.clip(mosaic_noisy, 0.0, 1.0)
    #linRGB = np.clip(linRGB, 0.0, 1.0)
    mosaic_blur = np.clip(mosaic_blur, 0.0, 1.0)

    return mosaic_noisy, mosaic_blur, linRGB, [sigma_s, sigma_r]



def crop_patch(img, meta, patch_size=(150, 150), stride=150, random_crop=False, burst_num=1):

    img_size = img.shape
    count = 0
    linRGB_list = []
    mosaic_blur_list = []
    print(img_size, patch_size)
    if patch_size[0]*2 < img_size[0] and patch_size[1]*2 < img_size[1]:
        if random_crop == True:
            crop_num = 1
            pos = [(np.random.randint(patch_size[1], img_size[1] - patch_size[1]),
                    np.random.randint(patch_size[0], img_size[0] - patch_size[1]))
                   for i in range(crop_num)]
        else:
            pos = [(x, y) for x in range(patch_size[1], img_size[1] - patch_size[1], stride) for y in
                   range(patch_size[0], img_size[0] - patch_size[0], stride)]

        for (xt, yt) in pos:
            cropped_img = img[yt - patch_size[0]:yt + patch_size[0], xt - patch_size[1]:xt + patch_size[1]]
            blur_burst = []
            for num in range(burst_num):
                img_mosaic_blur, img_linRGB = gen_blur(cropped_img, meta, kernel_size=[65,65])
                while ((img_mosaic_blur.shape[0] != patch_size[0]*2-64) | (img_mosaic_blur.shape[1] != patch_size[1]*2-64)):
                    print('shape is wrong !')
                    img_mosaic_blur, img_linRGB = gen_blur(cropped_img, meta, kernel_size=[65,65])
                blur_burst.append(img_mosaic_blur)

            linRGB_list.append(img_linRGB)
            mosaic_blur_list.append(np.stack(blur_burst, 0))
            count += 1
    elif patch_size[0]*2 == img_size[0] and patch_size[1]*2 == img_size[1]:
        blur_burst = []
        for num in range(burst_num):
            img_mosaic_blur, img_linRGB = gen_blur(img, meta, kernel_size=[65,65])
            while ((img_mosaic_blur.shape[0] != patch_size[0]*2-64) | (img_mosaic_blur.shape[1] != patch_size[1]*2-64)):
                print('shape is wrong !')
                img_mosaic_blur, img_linRGB = gen_blur(cropped_img, meta, kernel_size=[65,65])
            blur_burst.append(img_mosaic_blur)
        linRGB_list.append(img_linRGB)
        mosaic_blur_list.append(np.stack(blur_burst, 0))
    else:
        print('patch size is too large !')

    return mosaic_blur_list, linRGB_list

def gen_dataset(opt):
    src_path = opt['src_path']
    dst_path = opt['dst_path']
    dst_name = opt['dst_name']
    patch_size = opt['patch_size']
    crop_stride = opt['crop_stride']
    random_crop = opt['random_crop']
    burst_num = opt['burst_num']


    src_files = []
    for path in src_path:
        src_files.extend(sorted(glob.glob(path + "1*.ARW")))
        src_files.extend(sorted(glob.glob(path + "2*.ARW")))

    create_dir(dst_path)
    h5py_name = dst_path + dst_name
    h5f = h5py.File(h5py_name, 'w')

    random.shuffle(src_files)
    for i in range(len(src_files)):
        print(src_files[i])
        img_path = src_files[i]
        img_name = os.path.basename(img_path)
        file_name = img_name.split('.')[0]

        rawclass = rawpy.imread(img_path)
        raw = rawclass.raw_image_visible.astype(np.float32)

        meta = {}
        wbmuls = rawclass.camera_whitebalance
        meta['wb'] = [wbmuls[0]/wbmuls[1], 1.0, wbmuls[2]/wbmuls[1]]
        #print(meta['wb'])
        meta['XYZ2Cam'] = np.reshape(rawclass.rgb_xyz_matrix[0:3, :], [1, 9])[0]
        saturation, black = 16383, 512
        raw = (raw - black) / (saturation - black)
        raw = np.clip(raw, 0.0, 1.0)
        raw = raw[0:(raw.shape[0] // 4)*4, 0:(raw.shape[1] // 4)*4]
        img = MaxEntropy_Downsampling(raw)
        print(img.shape)
        if patch_size[0] == 0 or patch_size[1] == 0:
            patch_size[0], patch_size[1] = img.shape[0]-64, img.shape[1]-64
        mosaic_blur_list, linRGB_list = crop_patch(img, meta, (patch_size[0]/2+32, patch_size[1]/2+32),
                                        crop_stride, random_crop, burst_num)

        for num in range(len(linRGB_list)):

            mosaic_blur = mosaic_blur_list[num].copy()
            linRGB = linRGB_list[num].copy()


            mosaic_noisy, mosaic_blur, linRGB, sigma = process(mosaic_blur, linRGB, meta['wb'], burst_num)
            mosaic_noisy_list = []
            mosaic_blur_list = []
            for j in range(burst_num):
                mosaic_noisy_list.append(rggb2raw(mosaic_noisy[j]))
                mosaic_blur_list.append(rggb2raw(mosaic_blur[j]))
            mosaic_noisy = np.stack(mosaic_noisy_list, 0)
            mosaic_blur = np.stack(mosaic_blur_list, 0)

            g = h5f.create_group(str(i)+'_'+str(num))
            g.create_dataset('mosaic_noisy', shape=(burst_num, patch_size[0], patch_size[1], 1), data=mosaic_noisy)
            g.create_dataset('mosaic_blur', shape=(burst_num, patch_size[0], patch_size[1], 1), data=mosaic_blur)
            g.create_dataset('linRGB', shape=(patch_size[0], patch_size[1], 3), data=linRGB)
            g.create_dataset('wb', shape=(3,), data=meta['wb'])
            g.create_dataset('XYZ2Cam', shape=(9,), data=meta['XYZ2Cam'])
            g.create_dataset('sigma', shape=(2,), data=sigma)

    h5f.close()



if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    opt = {
            'src_path': ["/hdd/SID/Sony/long/",
                        ],
            'dst_path': "./Dataset/test/",
            'dst_name': "test.h5",
            'patch_size': [0,0],
            'crop_stride': 150,
            'random_crop': False,
            'burst_num': 4,
    }

    print("start...")
    gen_dataset(opt)
    print('end')
