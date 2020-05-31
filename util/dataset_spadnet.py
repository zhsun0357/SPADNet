""" SPADnet Dataset
"""

import torch
import torch.utils.data
import scipy.io
import numpy as np
import skimage.transform
import imageio
import re
import os

import pdb

class ToTensor(object):
    """Crop randomly the image in a sample."""

    def __init__(self):
        pass

    def __call__(self, sample):
        spad, depth_hr, intensity, mono_pred, mask, filename = \
                                                sample['spad'],\
                                                sample['depth_hr'],\
                                                sample['intensity'],\
                                                sample['mono_pred'],\
                                                sample['mask'],\
                                                sample['filename']

        sbr, photons = sample['sbr'], sample['photons']

        spad = torch.from_numpy(spad)
        depth_hr = torch.from_numpy(depth_hr)
        intensity = torch.from_numpy(intensity)
        mono_pred = torch.from_numpy(mono_pred)
        mask = torch.from_numpy(mask)

        return {'spad': spad, 'depth_hr': depth_hr,
                'intensity': intensity, 'sbr': sbr, 
                'photons': photons, 'mono_pred': mono_pred, 'mask': mask, 'filename': filename}


class RandomCrop(object):
    """Crop randomly the image in a sample."""

    def __init__(self, output_size, intensity_scale=1):
        self.output_size = output_size
        self.intensity_scale = intensity_scale

    def __call__(self, sample):
        spad, depth_hr, intensity, mono_pred, mask, filename = \
                                                sample['spad'],\
                                                sample['depth_hr'],\
                                                sample['intensity'],\
                                                sample['mono_pred'],\
                                                sample['mask'],\
                                                sample['filename']

        sbr, photons = sample['sbr'], sample['photons']

        h, w = spad.shape[2:]
        new_h = self.output_size
        new_w = self.output_size
        iscale = self.intensity_scale

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        spad = spad[:, :, top: top + new_h,
                    left: left + new_w]
        depth_hr = depth_hr.squeeze()[top: top + new_h,
                                    left: left + new_w]
        depth_hr = skimage.transform.resize(depth_hr,
                                           (self.output_size,
                                            self.output_size),
                                           mode='constant')
        depth_hr = depth_hr.reshape([1, self.output_size,
                                   self.output_size])
        intensity = intensity.squeeze()[top: top + new_h,
                                        left: left + new_w]
        intensity = skimage.transform.resize(intensity,
                                             (self.output_size,
                                              self.output_size),
                                             mode='constant')
        intensity = intensity.reshape([1, self.output_size,
                                       self.output_size])

        mono_pred = mono_pred.squeeze()[top: top + new_h,
                                        left: left + new_w]
        mono_pred = skimage.transform.resize(mono_pred,
                                             (self.output_size,
                                              self.output_size),
                                             mode='constant')
        mono_pred = mono_pred.reshape([1, self.output_size,
                                       self.output_size])

        mask = mask.squeeze()[top: top + new_h,
                              left: left + new_w]
        mask = skimage.transform.resize(mask,(self.output_size,
                                              self.output_size),
                                             mode='constant')
        mask = mask.reshape([1, self.output_size,
                             self.output_size])
        # pdb.set_trace()

        return {'spad': spad, 'depth_hr': depth_hr,
                'intensity': intensity, 'sbr': sbr, 
                'photons': photons, 'mono_pred': mono_pred, 'mask': mask, 'filename': filename}


class SpadDataset(torch.utils.data.Dataset):
    def __init__(self, datalist, noise_level, spad_datapath, mono_datapath, transform=None):

        self.nl = noise_level
        self.spad_datapath = spad_datapath
        self.mono_datapath = mono_datapath
        with open(datalist) as f:
            self.files_old = f.read().split()

        with open('blacklist.txt') as f:
            self.blacklist = f.read().split()

        self.spad_files = []
        ## check whether data is in blacklist and whether data has correct noise level
        for file in self.files_old:
            file_nl = re.search('nl\d+.mat', file).group()
            file_nl = file_nl.replace('nl', '')
            file_nl = int(file_nl.replace('.mat', ''))
            file_idx = file.replace('_nl{}.mat'.format(file_nl), '')

            if (not (file_idx in self.blacklist)) and (file_nl == self.nl):
                file = self.spad_datapath + file
                self.spad_files.append(file)
        self.transform = transform

    def __len__(self):
        return len(self.spad_files)

    def tryitem(self, idx):
        
        #--------------- check whether data is valid ----------------#
        mono_pred_file, mono_truth_file = self.get_mono_files(idx)
        
        ## if there is data corrupted or missing in dataset
        ## generate a blacklist file "blacklist_add.txt"
        while not os.path.exists(mono_pred_file):
            print(idx, ' ' + self.spad_files[idx] + '\n')

            with open('blacklist_add.txt', 'a+') as f:
                f.write( self.spad_files[idx] + '\n')
            self.spad_files.remove(self.spad_files[idx])
            mono_pred_file, mono_truth_file = self.get_mono_files(idx)

        #--------------- load spad measurements ----------------#
        # simulated spad measurements
        spad = np.asarray(scipy.sparse.csc_matrix.todense(scipy.io.loadmat(
            self.spad_files[idx])['spad'])).reshape([1, 512, 512, -1])
        spad = np.transpose(spad, (0, 3, 2, 1))

        intensity = np.asarray(scipy.io.loadmat(
            self.spad_files[idx])['intensity']).astype(
                np.float32).reshape([1, 512, 512])

        # high resolution depth maps
        depth_hr = (np.asarray(scipy.io.loadmat(
            self.spad_files[idx])['depth']).astype(
                np.float32).reshape([512, 512]))[None, :, :] / 12.276

        # sample metainfo
        sbr = np.asarray(scipy.io.loadmat(
            self.spad_files[idx])['SBR']).astype(np.float32)
        photons = np.asarray(scipy.io.loadmat(
            self.spad_files[idx])['mean_signal_photons']).astype(np.float32)

        #--------------- load monocular estimation ----------------#
        # monocular depth estimation (in png format)
        mono_pred = imageio.imread(mono_pred_file).astype(np.float64) / 25.5
        ## normalized to [0,1] range (original range: [0 10])
        mono_pred = mono_pred / 12.276

        mono_truth = imageio.imread(mono_truth_file).astype(np.float64) / 25.5
        ## normalized to [0,1] range (original range: [0 10])

        mono_pred = skimage.transform.resize(mono_pred,(512, 512), mode='constant')
        mono_truth = skimage.transform.resize(mono_truth,(512, 512), mode='constant')

        boolmask = (mono_truth >= 10.0) | (mono_truth <= 0.0)
        mask = 1. - boolmask.astype(np.float64)
        # mask out pixels with depth <= 0.0 or depth >= 10.0da

        filename = self.spad_files[idx]

        sample = {'spad': spad, 'depth_hr': depth_hr,
                  'intensity': intensity, 'sbr': sbr,
                  'photons': photons, 'mono_pred': mono_pred, 'mask': mask, 'filename': filename}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __getitem__(self, idx):
        try:
            sample = self.tryitem(idx)
        except Exception as e:
            print(idx, e)
            idx = idx + 1
            sample = self.tryitem(idx)
        return sample

    def get_mono_files(self, idx):

        file_idx = self.spad_files[idx].replace(self.spad_datapath, '')
        file_idx = file_idx.replace('_nl{}.mat'.format(self.nl), '')
        file_idx = file_idx.replace('spad_', '')

        mono_pred_file = self.mono_datapath + file_idx + '_pred.png'
        mono_truth_file = self.mono_datapath + file_idx + '_truth.png'

        return mono_pred_file, mono_truth_file
