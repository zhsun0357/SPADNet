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
