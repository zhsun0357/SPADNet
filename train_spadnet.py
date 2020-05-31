""" 
Training code for SPADNet
The code is adopted from SIGGRAPH 2018 paper by Lindell et al.:
https://www.computationalimaging.org/publications/single-photon-3d-imaging-with-deep-sensor-fusion/
"""

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter

import os
import argparse
from datetime import datetime
from shutil import copyfile
import skimage.io
from tqdm import tqdm

from util.dataset_spadnet import SpadDataset, RandomCrop, ToTensor
import configparser
from configparser import ConfigParser
from model_spadnet import ORLoss, SPADnet

import pdb

cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

# log-scale rebinning parameters
Linear_NUMBIN = 1024
NUMBIN = 128
Q = 1.02638 ## Solution for (q^128 - 1) / (q - 1) = 1024

parser = argparse.ArgumentParser(
        description='PyTorch Deep Sensor Fusion Training')
parser.add_argument('--option', default=None, type=str,
                    metavar='NAME', help='Name of model to use with options in config file, \
                    either SPADnet or LinearSPADnet')
## The code only support log scale rebinned SPADnet as model selection. Will make more 
## implementations available in the future

parser.add_argument('--logdir', default=None, type=str,
                    metavar='DIR', help='logging directory \
                    for logging')
parser.add_argument('--log_name', default=None, type=str,
                    metavar='DIR', help='name of tensorboard directory for this run\
                    for logging')
parser.add_argument('--config', default='config.ini', type=str,
                    metavar='FILE', help='name of configuration file')
parser.add_argument('--gpu', default=None, metavar='N',
                    help='which gpu')
parser.add_argument('--noise_param_idx', default=None, type = int,
                    help='which noise level we are training on (value 1-10)')
parser.add_argument('--lambda_tv', default=None, metavar='Float',
                    help='TV regularizer strength', type=float)
parser.add_argument('--batch_size', default=None, metavar='N',
                    help='minibatch size for optimization', type=int)
parser.add_argument('--workers', default=None, metavar='N',
                    help='number of dataloader workers', type=int)
parser.add_argument('--epochs', default=None, metavar='N',
                    help='number of epochs to train for', type=int)
parser.add_argument('--lr', default=None, metavar='Float',
                    help='learning rate', type=float)
parser.add_argument('--print_every', default=None, metavar='N',
                    help='Write to log every N iterations', type=int)
parser.add_argument('--save_every', default=None, metavar='N',
                    help='Save checkpoint every N iterations', type=int)
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--train_files', default=None, type=str, metavar='PATH',
                    help='path to list of train data')
parser.add_argument('--val_files', default=None, type=str, metavar='PATH',
                    help='path to list of validation data')
parser.add_argument('--override_ckpt_lr', default=None, action='store_true',
                    help='if resuming, override learning rate stored in\
                    checkpoint with command line/config file lr value')
parser.add_argument('--spad_datapath', default=None, type=str, metavar='PATH',
                    help='path to SPAD measurement data')
parser.add_argument('--mono_datapath', default=None, type=str, metavar='PATH',
                    help='path to monocular depth estimations')

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def tv(x):
    return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

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


def evaluate(model, val_loader, n_iter, model_name='SPADnet'):
    model.eval()
    sample = iter(val_loader).next()
    spad = sample['spad']
    intensity = sample['intensity']
    depth_hr = sample['depth_hr']
    # rates = sample['rates']
    mono_pred = sample['mono_pred']
    mask = sample['mask']
    filename = sample['filename']

    spad_var = Variable(spad.type(dtype))
    depth_var = Variable(depth_hr.type(dtype))
    intensity_var = Variable(intensity.type(dtype))
    # rates_var = Variable(rates.type(dtype))
    mono_pred_var = Variable(mono_pred.type(dtype))
    mask_var = Variable(mask.type(dtype))
    spad_var = tologscale(spad_var, NUMBIN, Q)
    rates_var = dmap2pc(depth_var, NUMBIN, Q, Linear_NUMBIN)
    rates_var = Variable(rates_var.type(dtype))
    mono_rates_var = dmap2pc(mono_pred_var, NUMBIN, Q, Linear_NUMBIN)
    mono_rates_var = Variable(mono_rates_var.type(dtype))

    denoise_out, sargmax = model(spad_var, mono_rates_var)
    denoise_out = denoise_out.unsqueeze(1)
    depth_var = depth_var * mask_var
    sargmax = sargmax * mask_var

    orloss = ORLoss(denoise_out, rates_var)

    writer.add_scalar('data/val_loss',
                      orloss.item(), n_iter)
    writer.add_scalar('data/val_rmse', np.sqrt(np.mean((
                      sargmax.data.cpu().numpy() -
                      depth_var.data.cpu().numpy())**2)) * 12.276, n_iter)

    if (n_iter % 500 == 0) or (n_iter == 1):

        im_est_depth = sargmax.data.cpu()[0:4, :, :, :].repeat(1, 3, 1, 1)
        im_depth_truth = depth_var.data.cpu()[0:4, :, :].repeat(1, 3, 1, 1)
        im_intensity = intensity_var.data.cpu()[0:4, :, :, :].repeat(1, 3, 1, 1)
        to_display = torch.cat((im_est_depth, im_depth_truth, im_intensity), 0)
        im_out = torchvision.utils.make_grid(to_display,
                                         normalize=True,
                                         scale_each=True,
                                         nrow=4)
        writer.add_image('image', im_out, n_iter)


def train(model, train_loader, val_loader, optimizer, n_iter,
          lambda_tv, epoch, logfile, val_every=5, save_every=10,
          model_name='SPADnet'):

    for sample in tqdm(train_loader):
        model.train()
        spad = sample['spad']
        # rates = sample['rates']
        intensity = sample['intensity']
        depth_hr = sample['depth_hr']
        mono_pred = sample['mono_pred']
        mask = sample['mask']

        spad_var = Variable(spad.type(dtype))
        depth_var = Variable(depth_hr.type(dtype))
        # rates_var = Variable(rates.type(dtype))
        intensity_var = Variable(intensity.type(dtype))
        mono_pred_var = Variable(mono_pred.type(dtype))
        mask_var = Variable(mask.type(dtype))
        spad_var = tologscale(spad_var, NUMBIN, Q)

        spad_var.requires_grad_(requires_grad = True)
        depth_var.requires_grad_(requires_grad = True)
        intensity_var.requires_grad_(requires_grad = True)
        mono_pred_var.requires_grad_(requires_grad = True)
        mask_var.requires_grad_(requires_grad = True)

        ## get ground truth denoised histogram (log scale rebinned) from ground truth depth
        rates_var = dmap2pc(depth_var, NUMBIN, Q, Linear_NUMBIN)
        rates_var = Variable(rates_var.type(dtype))
        rates_var.requires_grad_(requires_grad = True)
        
        ## get monocular estimation and up-project into 3D
        mono_rates_var = dmap2pc(mono_pred_var, NUMBIN, Q, Linear_NUMBIN)
        mono_rates_var = Variable(mono_rates_var.type(dtype))
        mono_rates_var.requires_grad_(requires_grad = True)

        # Run the model forward to compute scores and loss.
        denoise_out, sargmax = model(spad_var, mono_rates_var)
        denoise_out = denoise_out.unsqueeze(1)
        depth_var = depth_var * mask_var
        sargmax = sargmax * mask_var

        ## use OR loss and TV regularization
        tv_reg = lambda_tv * tv(sargmax)
        orloss = ORLoss(denoise_out, rates_var)
        loss = orloss + tv_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_iter += 1

        # log in tensorboard
        writer.add_scalar('data/train_loss',
                          orloss.item(), n_iter)
        writer.add_scalar('data/train_rmse', np.sqrt(np.mean((
                          sargmax.data.cpu().numpy() -
                          depth_var.data.cpu().numpy())**2)) * 12.276, n_iter)

        if (n_iter % val_every == 0) or (n_iter == 1):
            model.eval()
            evaluate(model, val_loader, n_iter, model_name)

        if n_iter % save_every == 0:
            save_checkpoint({
                'lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'n_iter': n_iter,
                 }, filename=logfile +
                 '/epoch_{}_{}.pth'.format(epoch, n_iter))

    return n_iter

def parse_arguments(args):
    opt = {}

    print('=> Reading config file and command line arguments')
    config = ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read(args.config)

    # figure out which model we're working with
    if args.option is not None:
        config.set('params', 'option', args.option)
    option = config.get('params', 'option')

    # handle the case of resuming an older model
    opt['resume_new_log_folder'] = False
    if args.resume:
        config.set(option, 'resume', args.resume)
    opt['resume'] = config.get(option, 'resume')
    if opt['resume']:
        if os.path.isfile(os.path.dirname(opt['resume']) + '/config.ini'):
            print('=> Resume flag set; switching to resumed config file')
            args.config = os.path.dirname(opt['resume']) + '/config.ini'
            config.clear()
            config._interpolation = configparser.ExtendedInterpolation()
            config.read(args.config)
        else:
            opt['resume_new_log_folder'] = True

    if args.gpu:
        config.set('params', 'gpu', args.gpu)
    if args.noise_param_idx:
        config.set('params', 'noise_param_idx', ' '.join(args.noise_param_idx))
    if args.logdir:
        config.set(option, 'logdir', args.logdir)
    if args.log_name:
        config.set(option, 'log_name', args.log_name)
    if args.batch_size:
        config.set(option, 'batch_size', str(args.batch_size))
    if args.workers:
        config.set(option, 'workers', str(args.workers))
    if args.epochs:
        config.set(option, 'epochs', str(args.epochs))
    if args.lambda_tv:
        config.set(option, 'lambda_tv', str(args.lambda_tv))
    if args.print_every:
        config.set(option, 'print_every', str(args.print_every))
    if args.save_every:
        config.set(option, 'save_every', str(args.save_every))
    if args.lr:
        config.set(option, 'lr', str(args.lr))
    if args.train_files:
        config.set(option, 'train_files', args.train_files)
    if args.val_files:
        config.set(option, 'val_files', args.val_files)

    # read all values from config file
    opt['lambda_tv'] = float(config.get(option, 'lambda_tv'))
    opt['intensity_scale'] = 1

    opt['gpu'] = config.get('params', 'gpu')
    opt['noise_param_idx'] = int(config.get('params', 'noise_param_idx'))
    opt['logdir'] = config.get(option, 'logdir')
    opt['log_name'] = config.get(option, 'log_name')
    opt['batch_size'] = int(config.get(option, 'batch_size'))
    opt['workers'] = int(config.get(option, 'workers'))
    opt['epochs'] = int(config.get(option, 'epochs'))
    opt['print_every'] = int(config.get(option, 'print_every'))
    opt['save_every'] = int(config.get(option, 'save_every'))
    opt['lr'] = float(config.get(option, 'lr'))
    opt['train_files'] = config.get(option, 'train_files')
    opt['val_files'] = config.get(option, 'val_files')
    opt['optimizer_init'] = config.get(option, 'optimizer')
    opt['model_name'] = config.get(option, 'model_name')
    opt['spad_datapath'] = config.get(option, 'spad_datapath')
    opt['mono_datapath'] = config.get(option, 'mono_datapath')
    # write these values to config file
    cfgfile = open(args.config, 'w')
    config.write(cfgfile)
    cfgfile.close()

    return opt


def main():
    # get arguments and modify config file as necessary
    args = parser.parse_args()
    opt = parse_arguments(args)
    # set gpu
    print('=> setting gpu to {}'.format(opt['gpu']))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu']

    # tensorboard log file
    global writer
    if not opt['resume'] or opt['resume_new_log_folder']:
        now = datetime.now()
        logfile = opt['logdir'] + '/' + opt['log_name'] + '_date_' + \
            now.strftime('%m_%d-%H_%M') + '/'
        writer = SummaryWriter(logfile)
        copyfile('./config.ini', logfile + 'config.ini')
    else:
        logfile = os.path.dirname(opt['resume'])
        writer = SummaryWriter(logfile)
    print('=> Tensorboard logging to {}'.format(logfile))

    model = eval(opt['model_name'] + '()')
    model.type(dtype)

    # initialize optimization tools
    params = filter(lambda p: p.requires_grad, model.parameters())
    if opt['optimizer_init'] != '':
        print('=> Loading optimizer from config...')
        optimizer = eval(opt['optimizer_init'])
    else:
        print('=> Using default Adam optimizer')
        optimizer = torch.optim.Adam(params, opt['lr'])
    
    # datasets and dataloader
    train_dataset = \
        SpadDataset(opt['train_files'], opt['noise_param_idx'], 
                    opt['spad_datapath'], opt['mono_datapath'],
                    transform=transforms.Compose(
                    [RandomCrop(128, intensity_scale=opt['intensity_scale']),
                     ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=opt['batch_size'],
                              shuffle=True, num_workers=opt['workers'],
                              pin_memory=True)
    val_dataset = \
        SpadDataset(opt['val_files'], opt['noise_param_idx'],
                    opt['spad_datapath'], opt['mono_datapath'],
                    transform=transforms.Compose(
                    [RandomCrop(128, intensity_scale=opt['intensity_scale']),
                     ToTensor()]))
    val_loader = DataLoader(val_dataset, batch_size=opt['batch_size'],
                            shuffle=True, num_workers=opt['workers'],
                            pin_memory=True)

    # resume checkpoint
    n_iter = 0
    start_epoch = 0
    if opt['resume']:
        if os.path.isfile(opt['resume']):
            print("=> loading checkpoint '{}'".format(opt['resume']))
            checkpoint = torch.load(opt['resume'])

            try:
                start_epoch = checkpoint['epoch']
            except KeyError as err:
                start_epoch = 0
                print('=> Can''t load start epoch, setting to zero')
            try:
                if not args.override_ckpt_lr:
                    opt['lr'] = checkpoint['lr']
                print('=> Loaded learning rate {}'.format(opt['lr']))
            except KeyError as err:
                print('=> Can''t load learning rate, setting to default')
            try:
                ckpt_dict = checkpoint['state_dict']
            except KeyError as err:
                ckpt_dict = checkpoint

            model_dict = model.state_dict()
            for k in ckpt_dict.keys():
                model_dict.update({k: ckpt_dict[k]})
            model.load_state_dict(model_dict)
            print('=> Loaded {}'.format(opt['resume']))

            try:
                optimizer_dict = optimizer.state_dict()
                ckpt_dict = checkpoint['optimizer']
                for k in ckpt_dict.keys():
                    optimizer_dict.update({k: ckpt_dict[k]})
                optimizer.load_state_dict(optimizer_dict)
            except (ValueError, KeyError) as err:
                print('=> Unable to resume optimizer from checkpoint')

            # set optimizer learning rate
            for g in optimizer.param_groups:
                g['lr'] = opt['lr']
            try:
                n_iter = checkpoint['n_iter']
            except KeyError:
                n_iter = 0

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opt['resume'], start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(opt['resume']))

    # run training epochs
    print('=> starting training')
    for epoch in range(start_epoch, opt['epochs']):
        print('epoch: {}, lr: {}'.format(epoch, optimizer.param_groups[0]['lr']))
        
        n_iter = train(model, train_loader, val_loader, optimizer, n_iter,
                       opt['lambda_tv'], epoch, logfile,
                       val_every=opt['print_every'],
                       save_every=opt['save_every'],
                       model_name=opt['model_name'])

        # decrease the learning rate
        for g in optimizer.param_groups:
            g['lr'] *= 0.5

        save_checkpoint({
            'lr': optimizer.param_groups[0]['lr'],
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'n_iter': n_iter,
             }, filename=logfile + '/epoch_{}_{}.pth'.format(epoch, n_iter))


if __name__ == '__main__':
    main()
