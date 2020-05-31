import torch
import torch.nn as nn
import numpy as np
import skimage.transform
from torch.autograd import Variable, Function
dtype = torch.cuda.FloatTensor

# log-scale rebinning parameters
Linear_NUMBIN = 1024
NUMBIN = 128
Q = 1.02638 ## Solution for (q^128 - 1) / (q - 1) = 1024


def ORLoss(denoise_out, rate, size_average=True):

    ## variant of ordinal regression loss

    denoise_hist = torch.nn.Softmax(dim=2)(denoise_out)
    denoise_cum = torch.cumsum(denoise_hist, dim = 2) + 1e-4
    denoise_logcum = torch.log(denoise_cum)
    ## log cumulative sum up, use a small offset to prevent numerical problem
    batchsize, _, numbin, H, W  = denoise_out.size()

    denoise_invcum = 1- denoise_cum + 3e-4
    # use small offset to prevent numerical problem
    denoise_loginvcum = torch.log(denoise_invcum)

    rate_cum = torch.cumsum(rate, dim=2)
    mask = (rate_cum > 0.5).float().cuda()
    invmask = (rate_cum < 0.5).float().cuda()
    loss = -(torch.sum(denoise_logcum * mask) + torch.sum(denoise_loginvcum * invmask))

    if size_average:
        loss =  loss / (batchsize * numbin * H*W)

    return loss


class SPADnet(nn.Module):
    def __init__(self):
        super(SPADnet, self).__init__()
        self.ds1 = nn.Sequential(
            nn.Conv3d(1, 1, 7, stride=2, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
        self.ds2 = nn.Sequential(
            nn.Conv3d(1, 1, 5, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
        self.ds3 = nn.Sequential(
            nn.Conv3d(1, 1, 3, stride=2, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(36, 36, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(36),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(28, 28, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(28),
            nn.ReLU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(16, 16, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.refine = nn.Sequential(
            nn.Conv3d(41, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.regress = nn.Sequential(
            nn.Conv3d(16, 1, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
        )

        self.conv0 = nn.Sequential(
            nn.Conv3d(1, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(1, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Conv3d(12, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Conv3d(12, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        # log-scale rebinning parameters
        self.linear_numbin = Linear_NUMBIN
        self.numbin = NUMBIN
        self.q = Q

    def inference(self, smax_denoise_out):
        
        ## 3D-2D projection with log scale
        bin_idx = np.arange(1, self.numbin + 1)
        dup = np.floor((np.power(self.q, bin_idx) - 1) / (self.q - 1)) / self.linear_numbin
        dlow = np.floor((np.power(self.q, bin_idx - 1) - 1) / (self.q - 1)) / self.linear_numbin
        dmid = torch.from_numpy((dup + dlow) / 2)

        dmid = dmid.squeeze().unsqueeze(1).unsqueeze(1).type(torch.cuda.FloatTensor)
        dmid.requires_grad_(requires_grad = True)

        weighted_smax = dmid * smax_denoise_out
        soft_argmax = weighted_smax.sum(1).unsqueeze(1) 
            
        return soft_argmax


    def forward(self, spad, mono_pc):

        # pass spad through U-net
        smax = torch.nn.Softmax2d()

        ds1_out = self.ds1(spad)
        ds2_out = self.ds2(ds1_out)
        ds3_out = self.ds3(ds2_out)

        conv0_out = self.conv0(spad)
        conv1_out = self.conv1(ds1_out)
        conv2_out = self.conv2(ds2_out)
        conv3_out = self.conv3(ds3_out)

        up3_out = self.up3(conv3_out)
        up2_out = self.up2(torch.cat((conv2_out, up3_out), 1))
        up1_out = self.up1(torch.cat((conv1_out, up2_out), 1))
        up0_out = torch.cat((conv0_out, up1_out), 1)

        refine_out = self.refine(torch.cat((mono_pc, up0_out), 1))
        regress_out = self.regress(refine_out)

        # squeeze and softmax for each-bin classification loss
        denoise_out = torch.squeeze(regress_out, 1)
        smax_denoise_out = smax(denoise_out)

        soft_argmax = self.inference(smax_denoise_out)

        return denoise_out, soft_argmax


