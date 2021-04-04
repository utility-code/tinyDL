if usegpu == True:
    import cupy as np
else:
    import numpy as np

from tinydl.layers import *
import matplotlib.pyplot as plt
import random

imtest = plt.imread("/home/eragon/Pictures/Wallpaper/drag.jpg")


def plotim(im):
    print(im.shape)
    plt.imshow(im)
    plt.axis("off")
    plt.show()


# CenterCrop

# CoarseDropout

# Crop

# CropAndPad

# CropNonEmptyMaskIfExists

# ElasticTransform

# Flip
def flip(im):
    return np.flip(im, random.choice([x for x in range(len(im.shape))]))


#  plotim(flip(imtest))
# GridDistortion

# GridDropout

# IAAAffine

# IAAPiecewiseAffine

# Lambda

# LongestMaxSize

# MaskDropout

# NoOp

# OpticalDistortion

# PadIfNeeded

# Perspective

# RandomCrop

# RandomCropNearBBox

# RandomGridShuffle

# RandomResizedCrop

# RandomRotate90

# RandomScale

# RandomSizedBBoxSafeCrop

# RandomSizedCrop

# Resize

# Rotate

# ShiftScaleRotate

# SmallestMaxSize

# Transpose

# Blur


def blur(im):
    k = np.array(
        [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]]
    )
    return conv2d(im[:, :, -1], k, pad=2)


# CLAHE

# ChannelDropout

# ChannelShuffle

# ColorJitter

# Downscale

# Emboss

# Equalize

# FDA

# FancyPCA

# FromFloat

# GaussNoise

# GaussianBlur

# GlassBlur

# HistogramMatching

# HueSaturationValue

# ISONoise

# ImageCompression

# InvertImg

# MedianBlur

# MotionBlur

# MultiplicativeNoise

# Normalize

# Posterize

# RGBShift

# RandomBrightnessContrast

# RandomFog

# RandomGamma

# RandomRain

# RandomShadow

# RandomSnow

# RandomSunFlare

# RandomToneCurve

# Sharpen


def sharpen(im):
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return conv2d(im[:, :, -1], k, pad=2)


# Solarize

# Superpixels

# ToFloat

# ToGray

# ToSepia
