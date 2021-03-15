import h5py
import matplotlib.image as mpimg
import os
import numpy as np
import glob
import math
import scipy.io as sio
import rawpy
from Dataset.preprocess import *


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process(raw, metadata):
    meta = {}
    meta['WhiteLevel'] = [metadata['meta']['saturation'][0][0][0][0]]
    meta['BlackLevel'] = [metadata['meta']['black'][0][0][0][0]]
    meta['Orientation'] = [metadata['meta']['orientation'][0][0][0][0]]
    meta['XYZ2Cam'] = metadata['meta']['xyz2cam'][0][0][0]
    #meta['pattern'] = metadata['meta']['cfapattern'][0][0][0]
    meta['wb'] = 1 / metadata['meta']['wb'][0][0][0]
    black = meta['BlackLevel'][0]
    saturation = meta['WhiteLevel'][0]
    raw = (raw - black) / (saturation - black)
    raw = np.clip(raw, 0.0, 1.0)
    raw = raw[0:(raw.shape[0] // 2)*2, 0:(raw.shape[1] // 2)*2]

    #if 'cfapattern' in metadata['meta']:
    if len(metadata['meta'].item()) == 6:
        pattern = metadata['meta']['cfapattern'][0][0][0]
        print(pattern)
        if pattern == 'BGGR':
            raw = raw[1:-1, 1:-1]
        elif pattern == 'GRBG':
            raw = raw[:, 1:-1]
        elif pattern == 'GBRG':
            raw = raw[1:-1, :]

    raw = raw[0:(raw.shape[0] // 4)*4, 0:(raw.shape[1] // 4)*4]
    img_linRGB = MaxEntropy_Downsampling(raw)

    return img_linRGB, meta


def crop_patch(img, meta, patch_size=(150, 150), stride=150, random_crop=False):

    img_size = img.shape
    count = 0
    linRGB_list = []
    mosaic_blur_list = []

    if random_crop == True:
        crop_num = 100
        pos = [(np.random.randint(patch_size[1], img_size[1] - patch_size[1]),
                np.random.randint(patch_size[0], img_size[0] - patch_size[0]))
               for i in range(crop_num)]
    else:
        pos = [(x, y) for x in range(patch_size[1], img_size[1] - patch_size[1], stride) for y in
               range(patch_size[0], img_size[0] - patch_size[0], stride)]

    for (xt, yt) in pos:
        cropped_img = img[yt - patch_size[0]:yt + patch_size[0], xt - patch_size[1]:xt + patch_size[1]]
        img_mosaic_blur, img_linRGB = gen_blur(cropped_img, meta, kernel_size=[65,65])
        while ((img_mosaic_blur.shape[0] != patch_size[0]*2-64) | (img_mosaic_blur.shape[1] != patch_size[1]*2-64)):
            print('shape is wrong !')
            img_mosaic_blur, img_linRGB = gen_blur(cropped_img, meta, kernel_size=[65,65])

        linRGB_list.append(img_linRGB)
        mosaic_blur_list.append(img_mosaic_blur)
        count += 1

    return mosaic_blur_list, linRGB_list

def gen_dataset(src_files, dst_path):
    create_dir(dst_path)
    h5py_name = dst_path + "train.h5"
    h5f = h5py.File(h5py_name, 'w')

    for i in range(len(src_files)):
        print(src_files[i])
        img_path = src_files[i]
        img_name = os.path.basename(img_path)
        file_name = img_name.split('.')[0]

        rawclass = rawpy.imread(img_path)
        raw = rawclass.raw_image_visible.astype(np.float32)
        metadata = sio.loadmat(os.path.dirname(img_path)+'/'+file_name+'.mat')
        img, meta = process(raw, metadata)

        mosaic_blur_list, linRGB_list = crop_patch(img, meta, (192, 192), 100, False)

        for num in range(len(linRGB_list)):

            mosaic_blur = mosaic_blur_list[num].copy()
            linRGB = linRGB_list[num].copy()

            g = h5f.create_group(str(i)+'_'+str(num))
            g.create_dataset('mosaic_blur', shape=(320, 320, 1), data=mosaic_blur)
            g.create_dataset('linRGB', shape=(320, 320, 3), data=linRGB)
            g.create_dataset('wb', shape=(3,), data=meta['wb'])
            g.create_dataset('XYZ2Cam', shape=(9,), data=meta['XYZ2Cam'])

    h5f.close()



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    src_path = ["/hdd4T_1/cm/codes/DataSet/fivek_dataset/perfect/RGGB/train/",
                "/hdd4T_1/cm/codes/DataSet/fivek_dataset/perfect/RGGB/train_supp/",
                "/hdd4T_1/cm/codes/DataSet/fivek_dataset/perfect/RGGB/train_supp2/",
                "/hdd4T_1/cm/codes/DataSet/fivek_dataset/perfect/GRBG/train/",
                "/hdd4T_1/cm/codes/DataSet/fivek_dataset/perfect/GBRG/train/",
                "/hdd4T_1/cm/codes/DataSet/fivek_dataset/perfect/BGGR/train/"]
    dst_path = "/hdd4T_1/cm/codes/DataSet/fivek_dataset/perfect/train_sv/"

    src_files = []
    for path in src_path:
        src_files.extend(sorted(glob.glob(path + "*.dng")))
    print("start...")
    gen_dataset(src_files, dst_path)
    print('end')
