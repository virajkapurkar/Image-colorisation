# Image-colorisation

A machine learning model to colorise grayscale images.

### Introduction

The aim is to build a machine learning model to automatically turn grayscale images into colored images. The model will be built from scratch (using PyTorch) i.e. without using any pre trained model.

At the end, the model will be able to colorize grayscale (or black and white) images.

### Overview

In image colorization, the goal is to produce a colored image given a grayscale input image. This problem is challenging because it is multimodal -- a single grayscale image may correspond to many plausible colored images. As a result, traditional models often relied on significant user input alongside a grayscale image.

#### The Problem

The aim is to infer a full-colored image, which has 3 values per pixel (lightness, saturation, and hue), from a grayscale image, which has only 1 value per pixel (lightness only).
To keep things simple for the time being, the model will only work with images of size $256 \times 256$, so our inputs are of size $256 \times 256 \times 1$ (the lightness channel) and our outputs are of size $256 \times 256 \times 2$ (the other two channels). In future, the aim is to improve the framework so that it can work for images of any resolution.

Rather than work with images in the RGB format, as people usually do, we will work with them in the [LAB colorspace](https://en.wikipedia.org/wiki/CIELAB_color_space) ($L$ightness, $A$, and $B$) . This colorspace contains exactly the same information as RGB, but it will make it easier for us to separate out the lightness channel from the other two (which we call $A$ and $B$). We'll try to predict the color values of the input image directly (we call this regression).

### The Model

Our model is a convolutional neural network. We first apply a number of convolutional layers to extract features from our image, and then we apply deconvolutional layers to upscale (increase the spacial resolution) of our features.

Specifically, the beginning of our model will be ResNet-18, an image classification network with 18 layers and residual connections. We will modify the first layer of the network so that it accepts grayscale input rather than colored input. Only first 6 layers of the ResNet-18 will be used for our purpose.

Note: At the end of the notebook, I've tried to implement a framework which can work with images of any size (with the same model). The output image will be of the same shape as that of input.

Intuition:

1. Use sliding window (window size 256 x 256) on the input image.
2. Make predictions on individual patches.
3. Take the average of the overlapping area.
4. Recombine the patches to form a complete image.
