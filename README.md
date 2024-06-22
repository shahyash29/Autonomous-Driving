# Segmentation

This repository contains a TensorFlow implementation of ENet (https://arxiv.org/pdf/1606.02147.pdf) based on the official Torch implementation (https://github.com/e-lab/ENet-training) and the Keras implementation by PavlosMelissinos (https://github.com/PavlosMelissinos/enet-keras). The model is trained on the Cityscapes dataset (https://www.cityscapes-dataset.com/).

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [Dataset](#dataset)
- [License](#license)

## Overview

This project implements ENet, a deep learning model for efficient and accurate semantic segmentation. The model has been trained on the Cityscapes dataset.

## Requirements

- tensorflow==1.15.0
- opencv-python
- numpy
- matplotlib
- tf_slim

## Installation

To set up the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/segmentation.git
    cd segmentation
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Preprocessing

Run `preprocess_data.py` to preprocess the Cityscapes dataset. Make sure that all Cityscapes training (validation) image directories are placed in `data/cityscapes/leftImg8bit/train` (`data/cityscapes/leftImg8bit/val`) and that all corresponding ground truth directories are placed in `data/cityscapes/gtFine/train` (`data/cityscapes/gtFine/val`).

```sh
python preprocess_data.py

Training
Run train.py to train the model. Make sure that preprocess_data.py has already been run.

Inference
Run run_on_sequence.py to run a model checkpoint on all frames in a Cityscapes demo sequence directory and create a video of the result. Set the model checkpoint path and sequence directory in the script before running.

Dataset
The Cityscapes dataset is used for training and validation. It contains high-quality pixel-level annotations of 5,000 frames collected in street scenes from 50 different cities. To download the dataset, visit the Cityscapes website.

Directory Structure
Place the dataset in the following directory structure:
segmentation/
├── data/
│   ├── cityscapes/
│   │   ├── leftImg8bit/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   ├── gtFine/
│   │   │   ├── train/
│   │   │   ├── val/
│   ├── mean_channels.pkl
│   ├── train_img_paths.pkl
│   ├── train_trainId_label_paths.pkl
│   ├── val_img_paths.pkl
│   ├── val_trainId_label_paths.pkl
