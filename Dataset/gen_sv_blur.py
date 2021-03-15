import numpy as np
import cv2
import math
from scipy.interpolate import interp1d
from imageio import imread, imwrite

def getkernel(kernel_size=31):
    LSnum = 10
    dxy = (np.random.rand(LSnum, 2) - 0.5) * np.random.randint(3, 10)
    down = 10
    xy = np.zeros([LSnum+1,2])
    for i in range(LSnum):
        xy[i+1,:] = xy[i,:] + dxy[i,:]
    xy_r = np.zeros([down*LSnum, 2])
    f = interp1d(np.arange(LSnum+1), xy[:,0], kind='cubic')
    xy_r[:,0] = f(np.linspace(0, LSnum, down*LSnum))
    f = interp1d(np.arange(LSnum+1),  xy[:,1], kind='cubic')
    xy_r[:,1] = f(np.linspace(0,LSnum,down*LSnum))


    [X, Y] = np.meshgrid(np.arange(kernel_size),np.arange(kernel_size))
    X = X - (kernel_size+1.)/2
    Y = Y - (kernel_size+1.)/2

    K = np.zeros([kernel_size, kernel_size])
    sigma = (np.random.rand() + 0.5) * 10
    for i in range(down*LSnum):
        K += np.exp(-((X - xy_r[i,0])**2 + (Y - xy_r[i,1])**2) / sigma)
    kmap = K / np.max(K) * 1.3
    kmap = np.expand_dims(kmap, axis=2)
    color_weight = np.random.rand(1,1,3)
    color_weight = color_weight/np.max(color_weight) + 1
    kmap = np.multiply(kmap, color_weight)
    #kmap = cv2.GaussianBlur(kmap, (5,5), 1)

    return kmap


def printlightstreaks(Img, ksize=31):
    [Height,Weight,Channel] = Img.shape
    LSnum = np.random.randint(0,7)
    Imgout = Img.copy()
    for i in range(LSnum):
        xc = np.random.randint(2*ksize,Height-2*ksize)
        yc = np.random.randint(2*ksize,Weight-2*ksize)
        k = getkernel(kernel_size=ksize)
        Imgout[xc:xc+ksize,yc:yc+ksize,:] += k
    #Imgout = np.clip(Imgout,0,1)
    return Imgout

def getRotattionMatirx(a, b, c):
#GETROTATIONMATRIX Summary of this function goes here
#Detailed explanation goes here
# R=[cos(b)*cos(c),cos(b)*sin(c),-sin(b);
#     sin(a)*sin(b)*cos(c)-cos(a)*sin(c),sin(a)*sin(b)*sin(c)+cos(a)*cos(c),sin(a)*cos(b);
#     cos(a)*sin(b)*cos(c)+sin(a)*sin(c),cos(a)*sin(b)*sin(c)-sin(a)*cos(c),cos(a)*cos(b)]
# R=R'
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(a), -math.sin(a)],
                   [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)],
                   [0, 1, 0],
                   [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0],
                    [math.sin(c), math.cos(c), 0],
                    [0, 0, 1]])
    R = Rz.dot(Ry.dot(Rx))
    return R

def getk3(y, x, xmax, ymax):
    Ns = len(x)
    #deta = 1
    #XX, YY = np.meshgrid(np.arange(-ymax, ymax+1), np.arange(-xmax, xmax+1))
    f = np.zeros((xmax*2+1, ymax*2+1, Ns))
    for i in range(Ns):
        #f[:,:,i] += 1/math.sqrt(0.4*math.pi)/deta * np.exp(-((XX-x[i])**2+(YY-y[i])**2)/0.4/deta**2)
        dy, dx = int(y[i]), int(x[i])
        f[dy+ymax+1,dx+xmax+1,i] += 1
    fout = np.sum(f, -1)

    fout = fout / np.sum(fout)

    return fout

def apply_matrix(img, matrix):
    r = (matrix[0, 0] * img[:, :, 0] + matrix[0, 1] * img[:, :, 1]
         + matrix[0, 2] * img[:, :, 2])
    g = (matrix[1, 0] * img[:, :, 0] + matrix[1, 1] * img[:, :, 1]
         + matrix[1, 2] * img[:, :, 2])
    b = (matrix[2, 0] * img[:, :, 0] + matrix[2, 1] * img[:, :, 1]
         + matrix[2, 2] * img[:, :, 2])
    results = np.stack((r, g, b), axis=-1)
    return results
#=============================================================================#

def gen_sv_psf(img, kernel_size=[63,63], k_sample=16):

    Height, Width, channel = img.shape
    [kh, kw] = kernel_size
    #=============================================================================#
    ## camera intinsics
    fx = 1700
    fy = 1700
    x0 = Width//2
    y0 = Height//2

    K = np.array([[fx, 0, x0],
                  [0, fy, y0],
                  [0, 0, 1]])
    invK = np.linalg.inv(K)
    # time
    #
    t_interval = np.random.uniform(0.01, 0.03) #interval time
    t_exposure = np.random.choice([0.125, 0.25, 0.5]) #exposure time
    dt = 0.001
    N = int((t_exposure + t_interval) / dt)       #sample interval
    N_init = int(t_interval / dt)
    #=================================================#
    down = 20
    gyro_max = 0.3 # max degree of gyro
    gyro_low = (np.random.rand(N//down, 3) - 0.5) * gyro_max
    gyro = np.zeros((N+2, 3))
    for i in range(3):
        f = interp1d(np.arange(N//down),  gyro_low[:,i], kind='cubic')
        gyro[:,i] = f(np.linspace(0,N//down-1,N+2))

    #gyro = (np.tile(np.random.rand(1,3),(N+2,1)) - 0.5) * gyro_max
    theta = np.zeros((N+2, 3))

    shift_max = 100 # max degree of shift
    vshift_low = (np.random.rand(N//down, 2) - 0.5) * shift_max
    vshift = np.zeros((N+2, 2))
    for i in range(2):
        f = interp1d(np.arange(N//down),  vshift_low[:,i], kind='cubic')
        vshift[:,i] = f(np.linspace(0,N//down-1,N+2))
    Tshift = np.zeros((N+2, 2))
    for i in range(N+1):
        theta[i+1, :] = theta[i, :] + gyro[i, :] * dt
        Tshift[i+1, :] = Tshift[i, :] + vshift[i, :] * dt

    theta = theta[1::, :]
    Tshift = Tshift[1::,:]

    #Tshift = np.stack([Tshift[:,0], Tshift[:,1], np.zeros(N+1)], -1)
    R = np.zeros((3, 3, N+1))
    for n in range(N+1):
        R[:,:,n] = getRotattionMatirx(theta[n, 0], theta[n, 1], theta[n, 2])
    #=======================================================#

    x, y = np.meshgrid(np.arange(0, Width//k_sample), np.arange(0, Height//k_sample))
    x, y = x * k_sample, y * k_sample

    x_ori, y_ori = np.meshgrid(np.arange(0, Width), np.arange(0, Height))

    UV = np.stack([x,y,np.ones([Height//k_sample, Width//k_sample])], -1)
    dUV = np.zeros((Height//k_sample, Width//k_sample, 2, N))
    blured = np.zeros(np.shape(img))
    for n in range(N_init, N):
        matrix = K.dot(R[:,:,n]).dot(invK)
        UVp = apply_matrix(UV, matrix)
        UVp = UVp[:,:,0:2] / np.stack([UVp[:,:,2], UVp[:,:,2]], -1)
        UVp[:,:,0] = UVp[:,:,0] + Tshift[n,0]
        UVp[:,:,1] = UVp[:,:,1] + Tshift[n,1]
        dUV[:,:,0,n] = UVp[:,:,0] - UV[:,:,0]
        dUV[:,:,1,n] = UVp[:,:,1] - UV[:,:,1]

        map_x = dUV[:,:,0,n].repeat(k_sample, 0).repeat(k_sample, 1).astype(np.float32)
        map_y = dUV[:,:,1,n].repeat(k_sample, 0).repeat(k_sample, 1).astype(np.float32)
        #mapxy = np.sqrt(map_x*map_x + map_y*map_y)
        #mapxy = (mapxy - np.min(mapxy)) / (np.max(mapxy)-np.min(mapxy))

        map_x = map_x + x_ori.astype(np.float32)
        map_y = map_y + y_ori.astype(np.float32)

        '''
        if n % 10 == 0:
            imwrite('map_x_'+str(n)+'.png', mapxy)
            imwrite('inter_'+str(n)+'.png', np.clip(cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)/1.5, 0, 1))
        '''
        blured += cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    blured = blured / (N - N_init)
    # center
    xc = int(np.mean(dUV[:,:,0,:]))
    yc = int(np.mean(dUV[:,:,1,:]))

    blured = blured[kh//2-yc:-kh//2+1-yc, kw//2-xc:-kw//2+1-xc, :]
    blured = np.clip(blured, 0, 1)
    #kernel = getk3(dUV[0,0,1,:], dUV[0,0,0,:], kernel_size[0]//2, kernel_size[1]//2)
    return blured, [xc, yc]

if __name__ == '__main__':

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    gt = imread('0005_clean.png').astype(np.float32)/255
    #gt = gt[:,:,0:3]
    gt = gt[0:640, 0:992]

    #gt = printlightstreaks(gt, ksize=63)

    gain = 1.3
    thr = 1
    mask = (gt[:,:,0]<thr) | (gt[:,:,1]<thr) | (gt[:,:,-1]<thr)
    mask = mask.astype(np.float)
    mask = np.tile(mask[:,:,np.newaxis], [1,1,3])
    gt = (gt / 30) * mask + (gt * gain) * (1 - mask)

    blurred, xyc = gen_sv_psf(np.clip(gt*30*1.5, 0, 3), kernel_size=[63,63], k_sample=16)
    imwrite('result.png', blurred)
    
