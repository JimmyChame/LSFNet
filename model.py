import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
try:
    from dcn.deform_conv import ModulatedDeformConvPack2 as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

#==============================================================================#
class ResBlock(nn.Module):

    def __init__(self, input_channel=3, output_channel=3):
        super().__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        if self.in_channel != self.out_channel:
            self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x):
        if self.in_channel != self.out_channel:
            x = self.conv0(x)
        conv1 = self.lrelu(self.conv1(x))
        conv2 = self.conv2(conv1)
        out = x + conv2
        return out
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

#============================================================================#
class Align_module(nn.Module):

    def __init__(self, channels=32, groups=8):
        super().__init__()

        self.conv_1 = nn.Conv2d(2*channels, channels, 1, 1)
        self.offset_conv1 = nn.Conv2d(channels, 32, 3, 1, 1)  # concat for diff
        self.offset_conv2 = nn.Conv2d(64, 32, 3, 1, 1)  # concat for offset
        self.offset_conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.dcnpack = DCN(channels, channels, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                            extra_offset_mask=True, offset_in_channel=32)
        self.up = nn.ConvTranspose2d(2*channels, channels, 2, 2)
        self.conv_2 = nn.Conv2d(2*channels, channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, ref_fea, nbf_fea, last_offset=None, last_fea=None):

        offset = torch.cat([ref_fea, nbf_fea], 1)
        offset = self.conv_1(offset)
        offset = self.lrelu(self.offset_conv1(offset))
        if last_offset is not None:
            last_offset = F.interpolate(last_offset, scale_factor=2, mode='bilinear', align_corners=False)
            offset = self.lrelu(self.offset_conv2(torch.cat([offset, last_offset * 2], dim=1)))
        offset = self.lrelu(self.offset_conv3(offset))
        out = self.lrelu(self.dcnpack([nbf_fea, offset]))
        if last_fea is not None:
            #last_fea = F.interpolate(last_fea, scale_factor=2, mode='bilinear', align_corners=False)
            last_fea = self.up(last_fea)
            out = self.conv_2(torch.cat([last_fea, out], 1))

        return out, offset


class Deghost_module(nn.Module):

    def __init__(self, channels=32):
        super().__init__()

        self.conv_1_1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv_1_2 = nn.Conv2d(channels, channels, 3, 1, 1)

        self.fusion = nn.Conv2d(2*channels, channels, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, ref_fea, nbf_fea):

        ref_fea = self.lrelu(self.conv_1_1(ref_fea))
        nbf_fea = self.lrelu(self.conv_1_2(nbf_fea))
        weight = torch.cat([ref_fea, nbf_fea], 1)
        weight = self.fusion(weight)
        weight = torch.sigmoid(weight)
        out = nbf_fea * weight

        return out

class LSFNet(nn.Module):

    def __init__(self, input_channel=4, output_channel=3, groups=8):
        super().__init__()

        self.conv_1_1 = nn.Conv2d(input_channel, 32, 3, 1, 1)
        self.conv_1_2 = nn.Conv2d(input_channel, 32, 3, 1, 1)
        self.Res_1_1 = ResBlock(32, 32)
        self.Res_1_2 = ResBlock(32, 32)

        self.align_1 = Align_module(32, groups)
        self.deghost_1 = Deghost_module(32)

        self.down_2_1 = nn.Conv2d(32, 64, 2, 2)
        self.down_2_2 = nn.Conv2d(32, 64, 2, 2)
        self.Res_2_1 = ResBlock(64, 64)
        self.Res_2_2 = ResBlock(64, 64)

        self.align_2 = Align_module(64, groups)
        self.deghost_2 = Deghost_module(64)

        self.down_3_1 = nn.Conv2d(64, 128, 2, 2)
        self.down_3_2 = nn.Conv2d(64, 128, 2, 2)
        self.Res_3_1 = ResBlock(128, 128)
        self.Res_3_2 = ResBlock(128, 128)

        self.align_3 = Align_module(128, groups)
        self.deghost_3 = Deghost_module(128)

        self.down_4_1 = nn.Conv2d(128, 256, 2, 2)
        self.down_4_2 = nn.Conv2d(128, 256, 2, 2)
        self.Res_4_1 = ResBlock(256, 256)
        self.Res_4_2 = ResBlock(256, 256)

        self.align_4 = Align_module(256, groups)
        self.deghost_4 = Deghost_module(256)
        self.fusion_4 = nn.Conv2d(512, 256, 1, 1)
        self.dres_4 = ResBlock(256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.fusion_3 = nn.Conv2d(128*3, 128, 1, 1)
        self.dres_3 = ResBlock(128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.fusion_2 = nn.Conv2d(64*3, 64, 1, 1)
        self.dres_2 = ResBlock(64, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.fusion_1 = nn.Conv2d(32*3, 32, 1, 1)
        self.dres_1 = ResBlock(32, 32)

        self.out = nn.Conv2d(32, output_channel*4, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, noisy, blur):

        ref_1 = self.conv_1_1(noisy)
        nbf_1 = self.conv_1_2(blur)
        ref_1 = self.Res_1_1(ref_1)
        nbf_1 = self.Res_1_2(nbf_1)

        ref_2 = self.lrelu(self.down_2_1(ref_1))
        nbf_2 = self.lrelu(self.down_2_2(nbf_1))
        ref_2 = self.Res_2_1(ref_2)
        nbf_2 = self.Res_2_2(nbf_2)

        ref_3 = self.lrelu(self.down_3_1(ref_2))
        nbf_3 = self.lrelu(self.down_3_2(nbf_2))
        ref_3 = self.Res_3_1(ref_3)
        nbf_3 = self.Res_3_2(nbf_3)

        ref_4 = self.lrelu(self.down_4_1(ref_3))
        nbf_4 = self.lrelu(self.down_4_2(nbf_3))
        ref_4 = self.Res_4_1(ref_4)
        nbf_4 = self.Res_4_2(nbf_4)

        nbf_4, offset_4 = self.align_4(ref_4, nbf_4)
        nbf_4 = self.deghost_4(ref_4, nbf_4)
        L4_fea = self.fusion_4(torch.cat([nbf_4, ref_4], 1))
        L4_fea = self.dres_4(L4_fea)

        nbf_3, offset_3 = self.align_3(ref_3, nbf_3, offset_4, nbf_4)
        nbf_3 = self.deghost_3(ref_3, nbf_3)
        L4_fea = self.up3(L4_fea)
        L3_fea = self.fusion_3(torch.cat([nbf_3, ref_3, L4_fea], 1))
        L3_fea = self.dres_3(L3_fea)

        nbf_2, offset_2 = self.align_2(ref_2, nbf_2, offset_3, nbf_3)
        nbf_2 = self.deghost_2(ref_2, nbf_2)
        L3_fea = self.up2(L3_fea)
        L2_fea = self.fusion_2(torch.cat([nbf_2, ref_2, L3_fea], 1))
        L2_fea = self.dres_2(L2_fea)

        nbf_1, offset_1 = self.align_1(ref_1, nbf_1, offset_2, nbf_2)
        nbf_1 = self.deghost_1(ref_1, nbf_1)
        L2_fea = self.up1(L2_fea)
        L1_fea = self.fusion_1(torch.cat([nbf_1, ref_1, L2_fea], 1))
        L1_fea = self.dres_1(L1_fea)

        out = self.out(L1_fea)
        out = self.pixel_shuffle(out)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                #torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.xavier_uniform_(m.weight.data)
                #torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
