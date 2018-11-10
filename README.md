# CV-Nanodegree-Projects
*Udacity Nanodegree - Computer Vision Projects*

This repository contains projects submitted for completing Udacity's [Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891).

## Projects

- **P1 - Facial Keypoints Detection**:  Build a facial keypoint detection system.
- **P2 - Image Captioning**: Captioning images from COCO dataset
- **P3 - Implementing SLAM**: Implementing SLAM algorithm for 2D motion and measurements

*Note:* The code has been slightly modified for running on local machine with GPU support.

### Hardware

- **OS**: Windows 10 Pro
- **Processor**: Intel Core i7-6700HQ CPU @ 2.60 GHz
- **Ram**: 16.0 GB
- **GPU**: GeForce GTX 960M

### Software

- Cuda Toolkit v9.2
- Anaconda

#### Anaconda Setup

1. Setup a conda environment with Python 3.6

```sh
conda create --name cv-nd-cuda92 python=3.6
activate cv-nd-cuda92
```

2. Install Pytorch GPU

```sh
conda install pytorch cuda92 -c pytorch
pip3 install torchvision
```

3. Install remaining packages

```sh
pip install -r requirements.txt
```