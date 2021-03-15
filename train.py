import os
import argparse
import numpy as np
import random, time
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import matplotlib.image as mpimg
from skimage.measure import compare_psnr, compare_ssim
from model import *
from dataloader import *
from Dataset.postprocess import *

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def step_lr_adjust(optimizer, epoch, init_lr=1e-4, step_size=20, gamma=0.1):
    lr = init_lr * gamma ** (epoch // step_size)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cycle_lr_adjust(optimizer, epoch, base_lr=1e-5, max_lr=1e-4, step_size=10, gamma=1):
    cycle = np.floor(1 + epoch/(2  * step_size))
    x = np.abs(epoch/step_size - 2 * cycle + 1)
    scale =  gamma ** (epoch // (2 * step_size))
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1-x)) * scale
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(opt):
    src_path =  opt['src_path']
    val_path = opt['val_path']
    print(src_path)
    print(val_path)
    ckpt_dir = opt['ckpt_dir']
    log_dir = opt['log_dir']
    patch_size = opt['patch_size']
    batch_size = opt['batch_size']
    n_epoch = opt['n_epoch']
    lr = opt['lr']
    milestone = opt['milestone']
    finetune = opt['finetune']
    init_epoch = opt['init_epoch']
    NetName = opt['NetName']
    t_loss = opt['train_loss']

    # Load dataset
    #dataset = Dataset_from_h5(src_path, patch_size=patch_size)
    #dataset_val = Dataset_h5_real(src_path=val_path, patch_size=320, train=False)
    #dataset = Dataset_from_h5_rgb(src_path, patch_size=patch_size)
    #dataset_val = Dataset_h5_real_rgb(src_path=val_path, patch_size=320, train=False)
    dataset = Dataset_from_h5_hdr(src_path, patch_size=patch_size)
    dataset_val = Dataset_from_h5_test(src_path=val_path)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=2, shuffle=False, num_workers=0, drop_last=True)
    # Build model
    model = NetName()
    model.initialize_weights()
    if finetune:
        model = torch.load(ckpt_dir+'model_%04d.pth' % init_epoch)
    init_epoch = init_epoch + 1

    if t_loss == 'L2':
        criterion = nn.MSELoss()
    elif t_loss == 'L1':
        criterion = nn.L1Loss()

    if torch.cuda.is_available():
        print(torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0]).cuda()
            criterion = criterion.cuda()
        else:
            model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=milestone, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    writer = SummaryWriter(log_dir)

    for epoch in range(init_epoch, n_epoch+1):

        loss_sum = 0
        step_lr_adjust(optimizer, epoch, init_lr=lr, step_size=milestone, gamma=0.5)
        print('Epoch {}, lr {}'.format(epoch, optimizer.param_groups[0]['lr']))
        start_time = time.time()
        for i, data in enumerate(dataloader):
            noisy, blur, label, Cam2sRGB = data
            if torch.cuda.is_available():
                noisy, blur, label, Cam2sRGB = noisy.cuda(), blur.cuda(), label.cuda(), Cam2sRGB.cuda()
            noisy, blur, label, Cam2sRGB = Variable(noisy), Variable(blur), Variable(label), Variable(Cam2sRGB)

            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            output = model(noisy, blur)
            #postprocess
            output = postprocess_torch(output, Cam2sRGB, hdr_compress=True)
            label = postprocess_torch(label, Cam2sRGB, hdr_compress=True)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

            if (i % 100 == 0) and (i != 0) :
                loss_avg = loss_sum / 100
                loss_sum = 0.0
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.8f} Time: {:4.4f}s".format(
                    epoch, n_epoch, i + 1, len(dataloader), loss_avg, time.time()-start_time))
                start_time = time.time()
                # Record train loss
                writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
                # Record learning rate
                #writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)
                writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
        # save model
        if epoch % 1 == 0:
            torch.save(model, os.path.join(ckpt_dir, 'model_%04d.pth' % (epoch)))

        # validation
        if epoch % 1 == 0:
            psnr = 0
            loss_val = 0
            model.eval()
            for i, data in enumerate(dataloader_val):
                noisy, blur, label, Cam2sRGB = data
                if torch.cuda.is_available():
                    noisy, blur, label, Cam2sRGB = noisy.cuda(), blur.cuda(), label.cuda(), Cam2sRGB.cuda()
                noisy, blur, label, Cam2sRGB = Variable(noisy), Variable(blur), Variable(label), Variable(Cam2sRGB)

                test_out = model(noisy, blur)
                test_out.detach_()

                #postprocess
                test_out = postprocess_torch(test_out, Cam2sRGB, hdr_compress=True)
                label = postprocess_torch(label, Cam2sRGB, hdr_compress=True)

                # 计算loss
                loss_val += criterion(test_out, label).item()
                rgb_out = test_out.cpu().numpy().transpose((0,2,3,1))
                clean = label.cpu().numpy().transpose((0,2,3,1))
                for num in range(rgb_out.shape[0]):
                    denoised = np.clip(rgb_out[num], 0, 1)
                    psnr += compare_psnr(clean[num], denoised)
            img_nums = rgb_out.shape[0] * len(dataloader_val)
            psnr = psnr / img_nums
            loss_val = loss_val / len(dataloader_val)
            print('Validating: {:0>3} , loss: {:.8f}, PSNR: {:4.4f}'.format(img_nums, loss_val, psnr))
            mpimg.imsave(ckpt_dir+"img/%04d_denoised.png" % epoch, denoised)
            writer.add_scalars('Loss_group', {'valid_loss': loss_val}, epoch)
            writer.add_scalar('valid_psnr', psnr, epoch)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    opt = {

        'src_path': "/ssd2T_3/multiexposure/train_2.h5",

        #'val_path': "/ssd2T_3/multiexposure/valid.h5",
        'val_path': "/ssd2T_3/multiexposure/test_2.h5",

        'ckpt_dir': "/hdd4T_1/cm/codes/multiexposure/ckpt/LSFNet_L1_hdr/",
        'log_dir': "/hdd4T_1/cm/codes/multiexposure/log/LSFNet_L1_hdr/",

        'batch_size': 16,
        'patch_size': 256,
        'n_epoch': 300,
        'milestone': 100,
        'lr': 1e-4,
        'finetune': False,
        'init_epoch':0,
        'NetName': LSFNet,
        'train_loss': 'L1',#'L2',
    }
    create_dir(opt['log_dir'])
    create_dir(opt['ckpt_dir'])
    create_dir(opt['ckpt_dir']+'img/')
    train(opt)
