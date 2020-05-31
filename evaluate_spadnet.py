import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import argparse
from tqdm import tqdm
import configparser
from configparser import ConfigParser
from model_spadnet import SPADnet
from util.dataset_spadnet import SpadDataset, ToTensor
from torchvision import transforms

import scipy
import scipy.io
import os
import pathlib

import pdb

cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

parser = argparse.ArgumentParser(
        description='PyTorch Deep Sensor Fusion NYUV2 Evaluation')
parser.add_argument('--option', default=None, type=str,
                    metavar='NAME', help='Name of model to use with options in config file, \
                    either SPADnet or LinearSPADnet')
## The code only support log scale rebinned SPADnet as model selection. Will make more 
## implementations available in the future

parser.add_argument('--config', default='val_config.ini', type=str,
                    metavar='FILE', help='name of configuration file')
parser.add_argument('--gpu', default=None, metavar='N',
                    help='which gpu')
parser.add_argument('--ckpt_noise_param_idx', default=None, type=str,
                    help='which noise level we are evaluating on \
                         (value 1-9)')
parser.add_argument('--test_files', default=None, type=str, metavar='PATH',
                    help='path to list of validation intensity files')
parser.add_argument('--out_datapath', default=None, type=str, metavar='PATH',
                    help='path to output results')
parser.add_argument('--spad_datapath', default=None, type=str, metavar='PATH',
                    help='path to SPAD measurement data')
parser.add_argument('--mono_datapath', default=None, type=str, metavar='PATH',
                    help='path to monocular depth estimations')

# log-scale rebinning parameters
Linear_NUMBIN = 1024
NUMBIN = 128
Q = 1.02638 ## Solution for (q^128 - 1) / (q - 1) = 1024


def tologscale(rates, numbin, q):

    ## convert pc to log scale (log rebinning)
    batchsize, _, _, H, W = rates.size()

    bin_idx = np.arange(1, numbin + 1)
    up = np.floor((np.power(q, bin_idx) - 1) / (q - 1))
    low = np.floor((np.power(q, bin_idx - 1) - 1) / (q - 1))

    log_rates = torch.zeros(batchsize, 1, numbin, H, W)
    for ii in range(numbin):
        log_rates[:,:,ii,:,:] = torch.sum(rates[:, :, int(low[ii]):int(up[ii]), :, :], dim = 2)

    return log_rates.cuda()


def dmap2pc(dmap, numbin, q, linear_numbin):

    ## 2D-3D up-projection
    bin_idx = np.arange(1, numbin + 1)
    dup = np.floor((np.power(q, bin_idx) - 1) / (q - 1)) / linear_numbin
    dlow = np.floor((np.power(q, bin_idx - 1) - 1) / (q - 1)) / linear_numbin
    dmid = (dup + dlow) / 2

    batchsize, _, H, W = dmap.size()

    rates = torch.zeros(batchsize, 1, numbin, H, W).cuda()
    for ii in np.arange(NUMBIN):
        rates[:,:,ii,:,:] = (dmap <= dup[ii]) & (dmap >= dlow[ii])
    rates = Variable(rates.type(dtype))
    rates.requires_grad_(requires_grad = True)
    
    return rates


def parse_arguments(args):
    config = ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.optionxform = str
    config.read(args.config)

    if args.option is not None:
        config.set('params', 'option', args.option)
    option = config.get('params', 'option')

    if args.gpu:
        config.set('params', 'gpu', args.gpu)
    if args.ckpt_noise_param_idx:
        config.set('params', 'ckpt_noise_param_idx',
                   ' '.join(args.ckpt_noise_param_idx))

    # read all values from config file
    opt = {}
    opt['gpu'] = config.get('params', 'gpu')
    
    if args.option is not None:
        config.set('params', 'option', args.option)
    option = config.get('params', 'option')
    opt['model_name'] = config.get(option, 'model_name')

    opt['ckpt_noise_param_idx'] = int(config.get('params', 'ckpt_noise_param_idx'))
    if not opt['ckpt_noise_param_idx']:
        opt['ckpt_noise_param_idx'] = np.arange(1, 11)

    opt['option'] = config.get('params', 'option')

    opt['checkpoint'] = []
    opt['checkpoint'].append(config.get(option, \
        'ckpt_noise_param_{}'.format(opt['ckpt_noise_param_idx'])))

    opt['test_files'] = config.get(option, 'test_files')
    opt['out_datapath'] = config.get(option, 'out_datapath')
    opt['spad_datapath'] = config.get(option, 'spad_datapath')
    opt['mono_datapath'] = config.get(option, 'mono_datapath')
    
    return opt

def get_output_file(datapath, filename):

    out_filename = filename.replace('.mat', '.npy')
    out_filename = out_filename.replace(datapath, '')
    out_filename = out_filename.replace('/spad', '')

    return out_filename


def process_denoise(opt, model, val_loader, n_iter, model_name='SPADnet'):
    
    print('=> Evaluating Model on NYUV2 Dataset...\n')

    if not os.path.exists(opt['out_datapath']):
        os.mkdir(opt['out_datapath'])

    for sample in tqdm(val_loader):

        spad = sample['spad']
        intensity = sample['intensity']
        depth = sample['depth_hr']
        mono_pred = sample['mono_pred']
        filename = sample['filename']

        spad_var = Variable(spad.type(dtype))
        depth_var = Variable(depth.type(dtype))
        intensity_var = Variable(intensity.type(dtype))
        mono_pred_var = Variable(mono_pred.type(dtype))

        batchsize, _, H, W = depth_var.size()
        mono_pred_var = mono_pred_var.view(batchsize, 1, H, W)

        ## patch size. Use 128x128 patches, with overlapping (64 step size)
        dim1 = 128
        dim2 = 128
        step = 64
        num_row = int(np.floor(H /step))
        num_col = int(np.floor(W /step))

        sargmax = torch.zeros(batchsize, H, W).cuda()

        for ii in range(num_row):
            for jj in range(num_col):

                spad_patch = spad_var[:, :, :, ii*step:(ii*step + dim1), jj*step:(jj*step + dim2)]
                mono_pred_patch = mono_pred_var[:, :, ii*step:(ii*step + dim1), jj*step:(jj*step + dim2)]
                spad_patch = tologscale(spad_patch, NUMBIN, Q)
                mono_rates_patch = dmap2pc(mono_pred_patch, NUMBIN, Q, Linear_NUMBIN)
                mono_rates_patch = Variable(mono_rates_patch.type(dtype))
                
                denoise_out, sargmax_patch = model(spad_patch, mono_rates_patch)
                
                beginx = 0 if (ii == 0) else step//2
                endx = dim1 if (ii == (num_row - 1)) else (dim1 - step//2)
                beginy = 0 if (jj == 0) else step//2
                endy = dim2 if (jj == (num_col - 1)) else (dim2 - step//2)

                sargmax[:, (ii*step + beginx):(ii*step + endx), (jj*step + beginy):(jj*step + endy)] = sargmax_patch[:, 0, beginx:endx, beginy:endy]

        for kk in range(batchsize):
            out_filename = get_output_file(opt['spad_datapath'], filename[kk])
            out_filename = opt['out_datapath'] + out_filename
            np.save(out_filename, sargmax[kk,17: H - 15, 17: W - 15].data.cpu().numpy())
            # crop out the boundary region
        n_iter += 1

    print('=> Evaluating Model on NYUV2 Dataset Finished\n')


def main():
    # get arguments and modify config file as necessary
    args = parser.parse_args()
    opt = parse_arguments(args)
    # set gpu
    print('=> setting gpu to {}'.format(opt['gpu']))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu']

    ################# Model loading #################
    model = eval(opt['model_name'] + '()')
    model.type(dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print('=> Loading checkpoint {}'.format(opt['checkpoint'][0]))
    ckpt = torch.load(opt['checkpoint'][0])
    model_dict = model.state_dict()
    try:
        ckpt_dict = ckpt['state_dict']
    except KeyError:
        print('Key error loading state_dict from checkpoint; assuming checkpoint contains only the state_dict')
        ckpt_dict = ckpt

    for k in ckpt_dict.keys():
        model_dict.update({k: ckpt_dict[k]})
    
    model.load_state_dict(model_dict)

    
    ################# Dataset loading #################
    val_dataset = \
    SpadDataset(opt['test_files'], opt['ckpt_noise_param_idx'],
                opt['spad_datapath'], opt['mono_datapath'],
                transform=transforms.Compose(
                [ToTensor()]))
    val_loader = DataLoader(val_dataset, batch_size=2,
                            shuffle=False, num_workers=1,
                            pin_memory=True)

    ################# Start evaluate #################
    n_iter = 0
    process_denoise(opt, model, val_loader, n_iter, model_name=opt['model_name'])


if __name__ == '__main__':
    main()
