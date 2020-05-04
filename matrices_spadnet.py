import numpy as np
import imageio
import skimage
import skimage.transform
import scipy
import scipy.io
import os
import re

import tqdm

#------------- Define different figures of merits -------------#

def delta(prediction, target, mask, threshold):
    """
    Given prediction and target, compute the fraction of indices i
    such that
    max(prediction[i]/target[i], target[i]/prediction[i]) < threshold
    """
    c = np.maximum(prediction[mask > 0]/target[mask > 0], target[mask > 0]/prediction[mask > 0])
    # print(c)
    if np.sum(mask) > 0:
        # print((c < threshold).astype(float))
        return np.sum((c < threshold).astype(float)) / (np.sum(mask))
    else:
        return 0.


def rel_abs_diff(prediction, target, mask, eps=1e-6):
    """
    The average relative absolute difference
    """
    diff = prediction - target
    out = np.sum(np.abs(diff[mask > 0])/(target[mask > 0] + eps))
    total = np.sum(mask).item()
    if total > 0:
        return (1. / np.sum(mask)) * out
    else:
        return 0.

def rel_sqr_diff(prediction, target, mask, eps=1e-6):
    """
    The average relative squared difference
    """
    diff = prediction - target
    out = np.sum((diff[mask > 0]**2)/(target[mask > 0] + eps))
    total = np.sum(mask).item()
    if total > 0:
        return (1. / np.sum(mask)) * out
    else:
        return 0.


def split_result(results):

    results_path = '/media/data3/szh/szh_validation_rapp/eval_log_testall_nl9_32_intensity_overlap_new_epoch5/'
    Pred = []

    for filename in results:
        check = re.match( r'pred_(.*)',filename, re.M|re.I)
        if check is not None:
            Pred.append(filename)

    return Pred


def calculate_matrices(results_path, truth_path, matrices_out):
    

    results = os.listdir(results_path)

    Pred = split_result(results)

    loss_fns = []
    loss_fns.append(("delta1", lambda p, t, m: delta(p, t, m, threshold=1.25)))
    loss_fns.append(("delta2", lambda p, t, m: delta(p, t, m, threshold=1.25 ** 2)))
    loss_fns.append(("delta3", lambda p, t, m: delta(p, t, m, threshold=1.25 ** 3)))
    loss_fns.append(("rel_abs_diff", rel_abs_diff))
    loss_fns.append(("rel_sqr_diff", rel_sqr_diff))
    
    total_losses = {loss_name: 0. for loss_name, _ in loss_fns}

    SE = 0.
    Num_pixels = 0.
    ii = 0

    print('=> Calculating Matrices...\n')
    for predfile in Pred:
        
        print(predfile + '\n')

        predfile = results_path + predfile
        depthfile = predfile.replace('pred', 'depth')
        depthfile = depthfile.replace(results_path, truth_path)
        maskfile = predfile.replace('pred', 'mask')
        maskfile = maskfile.replace(results_path, truth_path)


        pred = np.load(predfile) * 12.276 # convert to unit meter
        depth = np.load(depthfile) * 12.276 # convert to unit meter
        mask = np.load(maskfile)
        Num_pixels += np.sum(mask)
        SE += np.sum(((pred - depth) * mask)**2)

        with open(matrices_out, 'a+') as f:
            f.write('file: ' + predfile + '\n')

        for loss_name, loss_fn in loss_fns:
            loss = loss_fn(pred,depth,mask)
            total_losses[loss_name] += loss * np.sum(mask)
            with open(matrices_out, 'a+') as f:
                f.write(loss_name + ':' + str(loss) + '\n')

        with open(matrices_out, 'a+') as f:
            f.write('\n\n')

        ii += 1

    avg_losses = {loss_name: total_losses[loss_name]/Num_pixels for loss_name in total_losses}
    with open(matrices_out, 'a+') as f:
        f.write('AVG matrices:\n' + str(avg_losses))

    RMSE_avg = np.sqrt(SE / Num_pixels)
    with open(matrices_out, 'a+') as f:
        f.write('AVG RMSE: {}\n'.format(RMSE_avg))
    print('=> Calculating Matrices Finished\n')


if __name__ == "__main__":

    results_path = 'eval_spadnet_results/'
    truth_path = 'gt_nyuv2_depth/'
    matrices_out = 'matrices_spadnet.txt'

    calculate_matrices(results_path, truth_path, matrices_out)
