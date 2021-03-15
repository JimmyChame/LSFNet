# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 12:01:28 2019

@author: Chame
"""
import numpy as np
from Dataset.gen_sv_blur import gen_sv_psf, printlightstreaks

def MaxEntropy_Downsampling(raw):
    R = raw[0::2, 0::2]
    Gr = raw[0::2, 1::2]
    Gb = raw[1::2, 0::2]
    B = raw[1::2, 1::2]

    R = (R[0::2,0::2] + 3*R[0::2,1::2] + 3*R[1::2,0::2] + 9*R[1::2,1::2]) / 16
    Gr = (Gr[0::2,0::2] + Gr[0::2,1::2] + Gr[1::2,0::2] + Gr[1::2,1::2]) / 4
    Gb = (Gb[0::2,0::2] + Gb[0::2,1::2] + Gb[1::2,0::2] + Gb[1::2,1::2]) / 4
    B = (9*B[0::2,0::2] + 3*B[0::2,1::2] + 3*B[1::2,0::2] + B[1::2,1::2]) / 16
    G = (Gr + Gb) / 2

    RGB = np.stack([R,G,B], -1)
    return RGB

def inv_wb(img, gain, t=0.9):
    img_R, img_G, img_B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gr, gb = gain[0], gain[1]
    aR = (np.maximum(img_R-t, 0) / (1-t)) ** 2
    img_R = np.maximum(img_R/gr, (1-aR)*(img_R/gr) + aR*img_R)
    aB = (np.maximum(img_B-t, 0) / (1-t)) ** 2
    img_B = np.maximum(img_B/gb, (1-aB)*(img_B/gb) + aB*img_B)

    img_RGB = np.stack([img_R, img_G, img_B], -1)
    return img_RGB

def wb(img, gain):
    gr, gb = gain[0], gain[1]
    linear_R =  img[:, :, 0] * gr
    linear_G =  img[:, :, 1]
    linear_B =  img[:, :, 2] * gb
    img_RGB = np.stack([linear_R, linear_G, linear_B], -1)
    img_RGB = np.clip(img_RGB, 0.0, 1.0)
    return img_RGB

def mosaic_bayer(rgb):

    mosaic_img = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=rgb.dtype)
    mosaic_img[0::2, 0::2] = rgb[0::2, 0::2, 0]
    mosaic_img[0::2, 1::2] = rgb[0::2, 1::2, 1]
    mosaic_img[1::2, 0::2] = rgb[1::2, 0::2, 1]
    mosaic_img[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return mosaic_img

def gen_blur(img_linRGB, meta, kernel_size=[65,65]):

    kh, kw = kernel_size
    # color registration
    [gr, gg, gb] = meta['wb']
    int_RGB = wb(img_linRGB, [gr,gb])

    # random over-exposure
    if np.random.uniform(0, 1) < 0.01:
        int_RGB = printlightstreaks(int_RGB, ksize=65)

    gain = np.random.uniform(1.3, 2.3)
    mask = (int_RGB[:,:,0]<1) | (int_RGB[:,:,1]<1) | (int_RGB[:,:,-1]<1)
    mask = np.tile(mask[:,:,np.newaxis], [1,1,3])
    int_RGB = int_RGB * mask + (int_RGB * gain) * (1 - mask)

    # add spatial vatiant blur
    img_blur, xyc = gen_sv_psf(int_RGB, kernel_size=[kh,kw], k_sample=8)
    img_blur, int_RGB = np.clip(img_blur, 0, 1), np.clip(int_RGB, 0, 1)
    int_RGB  = int_RGB [kh//2-xyc[1]:-kh//2+1-xyc[1], kw//2-xyc[0]:-kw//2+1-xyc[0], :]
    # inv white balance
    img_linRGB = inv_wb(int_RGB, [gr,gb], t=0.9)
    img_blur = inv_wb(img_blur, [gr,gb], t=0.9)

    '''
    # add noise
    sigma_s = tf.random_uniform([1], 0.0, 0.001)
    sigma_r = tf.random_uniform([1], 0.0, 0.0001)
    #meta['sigma'] = tf.stack([sigma_s, sigma_r],-1)
    sigma_total = tf.sqrt(sigma_s * img_blur + sigma_r)
    img_blur = img_blur + tf.random_normal(img_blur.shape, stddev=sigma_total)
    img_blur = tf.clip_by_value(img_blur, 0.0, 1.0)
    '''
    # Mosaic
    img_mosaic_blur = mosaic_bayer(img_blur)

    return img_mosaic_blur, img_linRGB

def rggb2raw(rggb):
    [h,w,c] = rggb.shape
    raw = np.zeros([2*h,2*w])
    raw[0::2, 0::2] = rggb[:, :, 0]
    raw[0::2, 1::2] = rggb[:, :, 1]
    raw[1::2, 0::2] = rggb[:, :, 2]
    raw[1::2, 1::2] = rggb[:, :, 3]
    return raw

def raw2rggb(raw):
    raw_R = raw[0::2, 0::2]
    raw_Gr = raw[0::2, 1::2]
    raw_Gb = raw[1::2, 0::2]
    raw_B = raw[1::2, 1::2]
    rggb = np.stack([raw_R, raw_Gr, raw_Gb, raw_B], -1)
    return rggb

def pack_with_wb(raw, gr, gb):
    raw_R = raw[0::2, 0::2] * gr
    raw_Gr = raw[0::2, 1::2]
    raw_Gb = raw[1::2, 0::2]
    raw_B = raw[1::2, 1::2] * gb
    rggb = np.stack([raw_R, raw_Gr, raw_Gb, raw_B], -1)
    return rggb
