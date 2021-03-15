import os, random, glob
import numpy as np
import torch
import math
import torch.nn.functional as F
import torch.nn as nn


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

sRGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]])

def apply_cmatrix(img, matrix):
    # Applies CMATRIX to RGB input IM. Finds the appropriate weighting of the
    # old color planes to form the new color planes, equivalent to but much
    # more efficient than applying a matrix transformation to each pixel
    assert np.size(img, 2) == 3
    r = (matrix[0, 0] * img[:, :, 0] + matrix[0, 1] * img[:, :, 1]
         + matrix[0, 2] * img[:, :, 2])
    g = (matrix[1, 0] * img[:, :, 0] + matrix[1, 1] * img[:, :, 1]
         + matrix[1, 2] * img[:, :, 2])
    b = (matrix[2, 0] * img[:, :, 0] + matrix[2, 1] * img[:, :, 1]
         + matrix[2, 2] * img[:, :, 2])
    results = np.stack((r, g, b), axis=-1)

    return results

def get_ccm(XYZ2Cam):
    XYZ2Cam = np.reshape(XYZ2Cam, (3, 3))
    sRGB2Cam = XYZ2Cam.dot(sRGB2XYZ)
    sRGB2Cam = sRGB2Cam / np.tile(np.sum(sRGB2Cam, axis=1), [3, 1]).T
    Cam2sRGB = np.linalg.inv(sRGB2Cam)
    return Cam2sRGB

def postprocess(lin_rgb, XYZ2Cam, hdr_compress=False):
    lin_rgb = np.clip(lin_rgb, 0.0, 1.0)
    Cam2sRGB = get_ccm(XYZ2Cam)
    lin_srgb = apply_cmatrix(lin_rgb, Cam2sRGB)
    lin_srgb = np.clip(lin_srgb, 0.0, 1.0)

    #  gamma correction
    if hdr_compress:
        srgb_h = np.log(1+100*np.maximum(lin_srgb, 1e-8)) / np.log(1+100)
    else:
        srgb_h = pow(np.maximum(lin_srgb, 1e-8), 1/2.22)
    return srgb_h
#================== pytorch =================================#
def demosaic_bilinear(rggb, is_cuda=False):
    Batch, C, H, W = rggb.size()

    R = torch.zeros((Batch, 1, H*2, W*2))
    G = torch.zeros((Batch, 1, H*2, W*2))
    B = torch.zeros((Batch, 1, H*2, W*2))
    if is_cuda:
        R, G, B = R.cuda(), G.cuda(), B.cuda()

    R[:, :, 0::2, 0::2] = rggb[:, 0:1, :, :]
    G[:, :, 0::2, 1::2] = rggb[:, 1:2, :, :]
    G[:, :, 1::2, 0::2] = rggb[:, 2:3, :, :]
    B[:, :, 1::2, 1::2] = rggb[:, 3:4, :, :]

    R = F.pad(R, (1,1,1,1), 'reflect')
    G = F.pad(G, (1,1,1,1), 'reflect')
    B = F.pad(B, (1,1,1,1), 'reflect')

    F_G = np.array([[0.0, 1.0, 0.0],
                    [1.0, 4.0, 1.0],
                    [0.0, 1.0, 0.0]]) / 4
    F_RB = np.array([[1.0, 2.0, 1.0],
                    [2.0, 4.0, 2.0],
                    [1.0, 2.0, 1.0]]) / 4
    F_G = torch.FloatTensor(F_G).expand(1,1,3,3)
    F_RB = torch.FloatTensor(F_RB).expand(1,1,3,3)

    F_G = nn.Parameter(data=F_G, requires_grad=False)
    F_RB = nn.Parameter(data=F_RB, requires_grad=False)

    if is_cuda:
        F_G, F_RB = F_G.cuda(), F_RB.cuda()

    R = F.conv2d(R, F_RB, padding=0)
    G = F.conv2d(G, F_G, padding=0)
    B = F.conv2d(B, F_RB, padding=0)

    return torch.cat((R, G, B), 1)


def apply_ccms(images, ccms):
    """Applies color correction matrices."""
    images = images.permute(0, 2, 3, 1) # Permute the image tensor to BxHxWxC format from BxCxHxW format
    images = images[:, :, :, None, :]
    ccms   = ccms[:, None, None, :, :]
    outs   = torch.sum(images * ccms, dim=-1)
    outs   = outs.permute(0, 3, 1, 2)  # Re-Permute the tensor back to BxCxHxW format
    return outs

def gamma_compression(images, gamma=2.22):
    """Converts from linear to gamma space."""
    # Clamps to prevent numerical instability of gradients near zero.
    #images = images.permute(0, 2, 3, 1) # Permute the image tensor to BxHxWxC format from BxCxHxW format
    outs   = torch.clamp(images, min=1e-8) ** (1.0 / gamma)
    #outs   = outs.permute(0, 3, 1, 2)  # Re-Permute the tensor back to BxCxHxW format
    return outs


def postprocess_torch(images, Cam2sRGB, hdr_compress=False):
    """Processes a batch of camRGB images into sRGB images."""
    images = torch.clamp(images, min=0.0, max=1.0)
    # Color correction.
    images = apply_ccms(images, Cam2sRGB)
    # Gamma compression.
    images = torch.clamp(images, min=0.0, max=1.0)
    if hdr_compress:
        images = torch.log(1+100*torch.clamp(images, min=1e-8)) / math.log(1+100)
    else:
        images = gamma_compression(images)

    #images = torch.round(images * 255) / 255

    return images
