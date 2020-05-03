## SPADNet
[SPADnet: Deep RGB-SPAD Sensor Fusion Assisted by Monocular Depth Estimation]. 

[SPADnet: Deep RGB-SPAD Sensor Fusion Assisted by Monocular Depth Estimation]: https://www.osapublishing.org/oe/abstract.cfm?uri=oe-28-10-14948


Created by Zhanghao Sun, David Brian Lindell, [Olav Solgaard] and [Gordon Wetzstein] from Stanford University.

**We will make code and pre-trained model public soon.**

[Olav Solgaard]: https://solgaardlab.stanford.edu/#research
[Gordon Wetzstein]: http://www.computationalimaging.org

## Introduction
This repository is code release for our Optics Express paper "SPADnet: Deep RGB-SPAD Sensor Fusion Assisted by Monocular Depth Estimation". 

Construction of 3D scenes is essential for many applications. An emerging technique, single-photon-avalanche-diode (SPAD)-based 3D imaging, is attracting increasing attention. In this work, we aim at depth reconstruction from noisy SPAD measurements. Previous algorithms generally suffer from insufficient accuracy under low photon-flux conditions. 

In this paper, we introduce SPADnet—a neural network architecture for robust RGB-SPAD sensor fusion. We introduce monocular depth estimation into the network to assist the fusion process and also optimize the training procedure with a better objective function and log-scale rebinning. The proposed model achieves state-of-the-art performance in SPAD denoising. It is also more memory efficient and significantly faster than previous RGB-SPAD fusion approaches. This work can lead to the application of SPAD-based 3D reconstruction in broader contexts, such as 3D imaging with longer range and strong ambient light or 3D imaging on mobile devices. Also, the general insight that monocular depth estimator can facilitate fusion between 3D and 2D data modality may be helpful in other 3D sensing related scenarios.

For more details of our work, please refer to our technical paper.

## Citation
If you find our work useful in your research, please consider citing:


## Installation
We use Python 3.6 , Pytorch 1.0 and CUDA 9.0 for our experiments. One can install the conda environment from "environment.yml".

## Training
### Preparing data
We use [NYUV2] dataset for SPAD measurement simulation. We select out data with high quality (without large holes in ground truth depth map, with reasonable reflectivity value and so on), which are separated into training set, validation set and test set (10:1:1). Corresponding scene index are listed in "util/train_clean.txt", "util/val_clean.txt" and "util/test_clean.txt".

To generate SPAD measurement simulation, we generate 
We first generate monocular depth estimation .
We also provide [real-world captured SPAD data] for three different scenes.

We also provide a pre-trained snapshot of SPADnet model. It is trained for 5 epochs, which takes around 24hrs. You can find it [Here] (12.5MB). Note that this model dose not contain pre-trained monocular depth estimators.

[real-world captured SPAD data]: 
[NYUV2]: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
[Here]: https://drive.google.com/file/d/1KDzbRpT_HA0Y0lEtPabKKlUhgcQ40FWr/view?usp=sharing

## Evaluation
### Simulated Dataset
### Real-world Captured Dataset
