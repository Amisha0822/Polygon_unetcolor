Project Overview

This project implements a UNet model from scratch in PyTorch to colorize polygon images. The input is a black-and-white polygon image, and the model predicts the same polygon filled with a target color.

The main goal is to demonstrate semantic image-to-image translation on synthetic polygon data, which can later be extended to real-world segmentation and coloring tasks.


Motivation

Image colorization is a classical computer vision problem. While most works focus on natural images, this project focuses on synthetic polygon data to:

Explore UNet architecture and its skip connections.

Practice image-to-image learning tasks.

Build a fully functional pipeline for dataset generation, training, and inference.


Features

Custom synthetic polygon dataset generator

UNet model implemented from scratch

Training pipeline using PyTorch

Inference script for colorization

Experiment tracking using Weights & Biases- I did but not sure if it works when I show


Dataset

The dataset consists of synthetic polygon images:

Input: Binary polygon images (white polygons on a black background)

Target: Colored polygon images

Size: Adjustable, configurable in the dataset generator script

Dataset Generation

Random polygons with varying sides, positions, and sizes

Random color assignment

Output stored as RGB images for training

Model Architecture

The model is a UNet with encoder-decoder structure:

Encoder (Contracting path)

Multiple convolutional blocks (Conv → ReLU → BatchNorm → MaxPool)

Downsampling extracts spatial features

Decoder (Expanding path)

Upsampling blocks (Transpose Conv → ReLU → BatchNorm)

Skip connections from encoder layers to preserve fine details

Output Layer

3-channel RGB output




Future Work

Extend to real-world segmentation tasks

Experiment with different architectures like ResUNet

Add multiple polygon classes and semantic labeling

Use advanced losses (e.g., perceptual loss, SSIM) for better quality
