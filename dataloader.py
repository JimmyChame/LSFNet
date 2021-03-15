import glob
from PIL import Image
import numpy as np
import torch
import random
import h5py
import torch.utils.data as data
from Dataset.postprocess import *
#from Dataset.preprocess import mosaic_bayer
from Dataset.preprocess import *


class Dataset_from_h5(data.Dataset):

    def __init__(self, src_path, patch_size=128):

        self.path = src_path
        h5f = h5py.File(self.path, 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

        self.patch_size = patch_size

    def __getitem__(self, index):
        h5f = h5py.File(self.path, 'r')
        key = self.keys[index]

        g = h5f[key]
        mosaic_blur = np.array(g['mosaic_blur']).reshape(g['mosaic_blur'].shape)
        linRGB = np.array(g['linRGB']).reshape(g['linRGB'].shape)
        wb = np.array(g['wb']).reshape(g['wb'].shape)
        XYZ2Cam = np.array(g['XYZ2Cam']).reshape(g['XYZ2Cam'].shape)
        data = np.concatenate([mosaic_blur, linRGB], 2)
        h5f.close()

        # transfer
        p = 0.5
        if random.random() > p: #RandomRot90
            data = data.transpose(1, 0, 2)
        if random.random() > p: #RandomHorizontalFlip
            data = data[:, ::-1, :]
            data = data[:, 1:-1, :]
        if random.random() > p: #RandomVerticalFlip
            data = data[::-1, :, :]
            data = data[1:-1, :, :]

        (H, W, C) = data.shape
        rnd_h = random.randint(0, max(0, (H - self.patch_size)//2)) * 2
        rnd_w = random.randint(0, max(0, (W - self.patch_size)//2)) * 2
        patch = data[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

        #patch = np.clip(patch.astype(np.float32)/255.0, 0.0, 1.0)
        mosaic_blur = patch[:, :, 0]
        linRGB = patch[:, :, 1:4]

        #gain = np.random.uniform(1.3, 2.3)
        gain = random.uniform(1.3, 3)
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
        color_distortion_r = random.uniform(0.7, 0.9)
        color_distortion_b = random.uniform(0.7, 0.9)
        linRGB_short[:,:,0] = linRGB_short[:,:,0] * color_distortion_r
        linRGB_short[:,:,2] = linRGB_short[:,:,2] * color_distortion_b
        # nosie
        mosaic_noisy = mosaic_bayer(linRGB_short)
        sigma_s = random.uniform(0.00001, 0.001)
        sigma_r = random.uniform(0.01, 0.1) * sigma_s
        #meta['sigma'] = np.stack([sigma_s, sigma_r],-1)
        sigma_total = np.sqrt(sigma_s * mosaic_noisy + sigma_r)
        mosaic_noisy = mosaic_noisy + np.random.normal(scale=sigma_total, size=mosaic_noisy.shape)
        sigma_total = np.sqrt(sigma_s * mosaic_blur + sigma_r)
        mosaic_blur = mosaic_blur + np.random.normal(scale=sigma_total, size=mosaic_blur.shape)
        #print(sigma_r, sigma_s)

        mosaic_noisy = np.clip(mosaic_noisy, 0.0, 1.0)
        mosaic_blur = np.clip(mosaic_blur, 0.0, 1.0)
        # quantization
        mosaic_noisy = np.round(mosaic_noisy * (16383 - 512))  / (16383 - 512)
        mosaic_noisy = mosaic_noisy * ratio

        gr, gb = wb[0], wb[2]

        # Mosaic and wb
        raw_R = mosaic_blur[0::2, 0::2] * gr
        raw_Gr = mosaic_blur[0::2, 1::2]
        raw_Gb = mosaic_blur[1::2, 0::2]
        raw_B = mosaic_blur[1::2, 1::2] * gb
        mosaic_blur = np.stack([raw_R, raw_Gr, raw_Gb, raw_B], -1)

        raw_R = mosaic_noisy[0::2, 0::2] * gr
        raw_Gr = mosaic_noisy[0::2, 1::2]
        raw_Gb = mosaic_noisy[1::2, 0::2]
        raw_B = mosaic_noisy[1::2, 1::2] * gb
        mosaic_noisy = np.stack([raw_R, raw_Gr, raw_Gb, raw_B], -1)

        linear_R =  linRGB[:, :, 0] * gr
        linear_G =  linRGB[:, :, 1]
        linear_B =  linRGB[:, :, 2] * gb
        linRGB = np.stack([linear_R, linear_G, linear_B], -1)

        mosaic_noisy = np.clip(mosaic_noisy, 0.0, 1.0)
        linRGB = np.clip(linRGB, 0.0, 1.0)
        mosaic_blur = np.clip(mosaic_blur, 0.0, 1.0)

        #mosaic_noisy = np.clip(mosaic_noisy/gain, 0.0, 1.0)
        #linRGB = np.clip(linRGB/gain, 0.0, 1.0)
        #mosaic_blur = np.clip(mosaic_blur, 0.0, 1.0)/gain


        Cam2sRGB = get_ccm(XYZ2Cam)
        Cam2sRGB  = torch.FloatTensor(Cam2sRGB)

        mosaic_noisy = torch.from_numpy(np.ascontiguousarray(np.transpose(mosaic_noisy, (2, 0, 1)))).float()
        mosaic_blur = torch.from_numpy(np.ascontiguousarray(np.transpose(mosaic_blur, (2, 0, 1)))).float()
        linRGB = torch.from_numpy(np.ascontiguousarray(np.transpose(linRGB, (2, 0, 1)))).float()

        return mosaic_noisy, mosaic_blur, linRGB, Cam2sRGB

    def __len__(self):
        return len(self.keys)

class Dataset_h5_real(data.Dataset):

    def __init__(self, src_path, patch_size=128, train=True):
        if train:
            self.path = src_path
        else:
            self.path = src_path
        h5f = h5py.File(self.path, 'r')
        self.keys = list(h5f.keys())
        if train:
            random.shuffle(self.keys)
        h5f.close()

        self.patch_size = patch_size
        self.train = train

    def __getitem__(self, index):
        h5f = h5py.File(self.path, 'r')
        key = self.keys[index]

        g = h5f[key]
        mosaic_noisy = np.array(g['mosaic_noisy']).reshape(g['mosaic_noisy'].shape)
        mosaic_blur = np.array(g['mosaic_blur']).reshape(g['mosaic_blur'].shape)
        linRGB = np.array(g['linRGB']).reshape(g['linRGB'].shape)
        wb = np.array(g['wb']).reshape(g['wb'].shape)
        XYZ2Cam = np.array(g['XYZ2Cam']).reshape(g['XYZ2Cam'].shape)
        if len(mosaic_noisy.shape) == 4:
            mosaic_noisy = mosaic_noisy[-1]
        if len(mosaic_blur.shape) == 4:
            mosaic_blur = mosaic_blur[-1]
        data = np.concatenate([mosaic_noisy, mosaic_blur, linRGB], 2)
        h5f.close()

        if self.train:
            p = 0.5
            if random.random() > p: #RandomRot90
                data = data.transpose(1, 0, 2)
            if random.random() > p: #RandomHorizontalFlip
                data = data[:, ::-1, :]
                data = data[:, 1:-1, :]
            if random.random() > p: #RandomVerticalFlip
                data = data[::-1, :, :]
                data = data[1:-1, :, :]

            (H, W, C) = data.shape
            rnd_h = random.randint(0, max(0, (H - self.patch_size)//2)) * 2
            rnd_w = random.randint(0, max(0, (W - self.patch_size)//2)) * 2
            patch = data[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        else:
            patch = data

        mosaic_noisy = patch[:, :, 0]
        mosaic_blur = patch[:, :, 1]
        linRGB = patch[:, :, 2:5]

        mosaic_noisy = np.clip(mosaic_noisy, 0.0, 1.0)
        mosaic_noisy = raw2rggb(mosaic_noisy)
        mosaic_blur = raw2rggb(mosaic_blur)

        Cam2sRGB = get_ccm(XYZ2Cam)
        Cam2sRGB  = torch.FloatTensor(Cam2sRGB)

        mosaic_noisy = torch.from_numpy(np.ascontiguousarray(np.transpose(mosaic_noisy, (2, 0, 1)))).float()
        mosaic_blur = torch.from_numpy(np.ascontiguousarray(np.transpose(mosaic_blur, (2, 0, 1)))).float()
        linRGB = torch.from_numpy(np.ascontiguousarray(np.transpose(linRGB, (2, 0, 1)))).float()

        return mosaic_noisy, mosaic_blur, linRGB, Cam2sRGB

    def __len__(self):
        return len(self.keys)

class Dataset_from_h5_test(data.Dataset):

    def __init__(self, src_path):

        self.path = src_path
        h5f = h5py.File(self.path, 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

    def __getitem__(self, index):
        h5f = h5py.File(self.path, 'r')
        key = self.keys[index]

        g = h5f[key]
        mosaic_noisy = np.array(g['mosaic_noisy']).reshape(g['mosaic_noisy'].shape)
        mosaic_blur = np.array(g['mosaic_blur']).reshape(g['mosaic_blur'].shape)
        linRGB = np.array(g['linRGB']).reshape(g['linRGB'].shape)
        wb = np.array(g['wb']).reshape(g['wb'].shape)
        XYZ2Cam = np.array(g['XYZ2Cam']).reshape(g['XYZ2Cam'].shape)
        h5f.close()

        mosaic_noisy = mosaic_noisy[0, 0:(linRGB.shape[0]//16)*16, 0:(linRGB.shape[1]//16)*16, 0] # first one
        mosaic_blur = mosaic_blur[0, 0:(linRGB.shape[0]//16)*16, 0:(linRGB.shape[1]//16)*16, 0] # first one
        linRGB = linRGB[0:(linRGB.shape[0]//16)*16, 0:(linRGB.shape[1]//16)*16]
        mosaic_noisy = np.clip(mosaic_noisy, 0.0, 1.0)
        mosaic_blur = np.clip(mosaic_blur, 0.0, 1.0)
        linRGB = np.clip(linRGB, 0.0, 1.0)

        mosaic_noisy = raw2rggb(mosaic_noisy)
        mosaic_blur = raw2rggb(mosaic_blur)

        Cam2sRGB = get_ccm(XYZ2Cam)
        Cam2sRGB  = torch.FloatTensor(Cam2sRGB)

        mosaic_noisy = torch.from_numpy(np.ascontiguousarray(np.transpose(mosaic_noisy, (2, 0, 1)))).float()
        mosaic_blur = torch.from_numpy(np.ascontiguousarray(np.transpose(mosaic_blur, (2, 0, 1)))).float()
        linRGB = torch.from_numpy(np.ascontiguousarray(np.transpose(linRGB, (2, 0, 1)))).float()

        return mosaic_noisy, mosaic_blur, linRGB, Cam2sRGB

    def __len__(self):
        return len(self.keys)
#============================================================================================================================#

class Dataset_from_h5_hdr(data.Dataset):

    def __init__(self, src_path, patch_size=128):

        self.path = src_path
        h5f = h5py.File(self.path, 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

        self.patch_size = patch_size

    def __getitem__(self, index):
        h5f = h5py.File(self.path, 'r')
        key = self.keys[index]

        g = h5f[key]
        mosaic_blur = np.array(g['mosaic_blur']).reshape(g['mosaic_blur'].shape)
        linRGB = np.array(g['linRGB']).reshape(g['linRGB'].shape)
        wb = np.array(g['wb']).reshape(g['wb'].shape)
        XYZ2Cam = np.array(g['XYZ2Cam']).reshape(g['XYZ2Cam'].shape)
        data = np.concatenate([mosaic_blur, linRGB], 2)
        h5f.close()

        # transfer
        p = 0.5
        if random.random() > p: #RandomRot90
            data = data.transpose(1, 0, 2)
        if random.random() > p: #RandomHorizontalFlip
            data = data[:, ::-1, :]
            data = data[:, 1:-1, :]
        if random.random() > p: #RandomVerticalFlip
            data = data[::-1, :, :]
            data = data[1:-1, :, :]

        (H, W, C) = data.shape
        rnd_h = random.randint(0, max(0, (H - self.patch_size)//2)) * 2
        rnd_w = random.randint(0, max(0, (W - self.patch_size)//2)) * 2
        patch = data[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

        #patch = np.clip(patch.astype(np.float32)/255.0, 0.0, 1.0)
        mosaic_blur = patch[:, :, 0]
        linRGB = patch[:, :, 1:4]

        #gain = random.uniform(1.3, 2.3)
        gain = random.uniform(1.3, 3)
        mosaic_blur = mosaic_blur * gain
        linRGB = linRGB * gain

        ratio = 30
        thr = gain
        mask = (linRGB[:,:,0]<thr) | (linRGB[:,:,1]<thr) | (linRGB[:,:,-1]<thr)
        mask = mask.astype(np.float)
        mask = np.tile(mask[:,:,np.newaxis], [1,1,3])
        linRGB_short = (linRGB / ratio) * mask + linRGB * (1 - mask)

        # dark current
        #dark_noise = random.uniform(0.01, 0.03) / ratio
        #mosaic_noisy = mosaic_noisy - dark_noise
        color_distortion_r = random.uniform(0.7, 0.9)
        color_distortion_b = random.uniform(0.7, 0.9)
        linRGB_short[:,:,0] = linRGB_short[:,:,0] * color_distortion_r
        linRGB_short[:,:,2] = linRGB_short[:,:,2] * color_distortion_b
        # nosie
        mosaic_noisy = mosaic_bayer(linRGB_short)
        sigma_s = random.uniform(0.00001, 0.001)
        sigma_r = random.uniform(0.01, 0.1) * sigma_s
        #meta['sigma'] = np.stack([sigma_s, sigma_r],-1)
        sigma_total = np.sqrt(sigma_s * mosaic_noisy + sigma_r)
        mosaic_noisy = mosaic_noisy + np.random.normal(scale=sigma_total, size=mosaic_noisy.shape)
        sigma_total = np.sqrt(sigma_s * mosaic_blur + sigma_r)
        mosaic_blur = mosaic_blur + np.random.normal(scale=sigma_total, size=mosaic_blur.shape)
        #print(sigma_r, sigma_s)

        mosaic_noisy = np.clip(mosaic_noisy, 0.0, 1.0)
        mosaic_blur = np.clip(mosaic_blur, 0.0, 1.0)
        # quantization
        mosaic_noisy = np.round(mosaic_noisy * (16383 - 512))  / (16383 - 512)
        mosaic_noisy = mosaic_noisy * ratio

        gr, gb = wb[0], wb[2]

        # Mosaic and wb
        raw_R = mosaic_blur[0::2, 0::2] * gr
        raw_Gr = mosaic_blur[0::2, 1::2]
        raw_Gb = mosaic_blur[1::2, 0::2]
        raw_B = mosaic_blur[1::2, 1::2] * gb
        mosaic_blur = np.stack([raw_R, raw_Gr, raw_Gb, raw_B], -1)

        raw_R = mosaic_noisy[0::2, 0::2] * gr
        raw_Gr = mosaic_noisy[0::2, 1::2]
        raw_Gb = mosaic_noisy[1::2, 0::2]
        raw_B = mosaic_noisy[1::2, 1::2] * gb
        mosaic_noisy = np.stack([raw_R, raw_Gr, raw_Gb, raw_B], -1)

        linear_R =  linRGB[:, :, 0] * gr
        linear_G =  linRGB[:, :, 1]
        linear_B =  linRGB[:, :, 2] * gb
        linRGB = np.stack([linear_R, linear_G, linear_B], -1)

        mosaic_noisy = np.clip(mosaic_noisy/gain, 0.0, 1.0)
        linRGB = np.clip(linRGB/gain, 0.0, 1.0)
        mosaic_blur = np.clip(mosaic_blur, 0.0, 1.0)/gain


        Cam2sRGB = get_ccm(XYZ2Cam)
        Cam2sRGB  = torch.FloatTensor(Cam2sRGB)

        mosaic_noisy = torch.from_numpy(np.ascontiguousarray(np.transpose(mosaic_noisy, (2, 0, 1)))).float()
        mosaic_blur = torch.from_numpy(np.ascontiguousarray(np.transpose(mosaic_blur, (2, 0, 1)))).float()
        linRGB = torch.from_numpy(np.ascontiguousarray(np.transpose(linRGB, (2, 0, 1)))).float()

        return mosaic_noisy, mosaic_blur, linRGB, Cam2sRGB

    def __len__(self):
        return len(self.keys)
