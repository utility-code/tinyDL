# TinyDL

- Tiny Deep learning library (+ GPU support!!!)
- Have a look at what works so far :)
- New features will be added as time goes
- Only needs numpy and pycuda for the library

## Requirements
- numpy
- matplotlib
- pycuda  (for GPU)

## How to run
- Configure parameters in config.py
- python main.py

## Why?
- Pytorch is too complicated to learn from. (Please. I tried. There are a million folders. My brain >.<)
- This does not intend to be Pytorch. Just to understand what goes into it
- An attempt at recreating most of the essential components from scratch
- Eventual blogs on it as well

## Whats there so far
### Unified config file
- Okay it took me 2 minutes to add but I am proud of it
### Experiment logging
- Wow I am actually productive today xD
- Define a directory
- It saves experiments with their losses over epochs along with a representation of the model (somewhat xD)
- auto increment
### Plots
- Accuracy/Loss plot
### Norm
- Dropout
### Complete training
- Forward/backprop
- Loss
- Gradient Descent
### Basic dataloader
- WIP
- Supports multiprocessing
- Accepts int, float, numpy arrays
- Note : No tensors yet
### Layers/Activations
- linear
- sigmoid
- relu
- tanh
### Augmentations
- Only works for 2d images (cries)
- Note that this is a super work in progress xD
- They will only work with images/other stuff for now. No masks/bboxes/keypoints
- Random flip
- sharpen
### Progress bar
- A progress bar generator xD (I was going to use tqdm. But well.)
- If you have a small screen/terminal window, reduce the length parameter
### Loss functions
- MSE

## Lazy list on what to add
- (Just adding stuff here when I notice something. Not exhaustive and prone to huge changes)j
- conv as part of training
- batchnorm
- Dropout
- pooling
- dilated conv
- Avoid using hardcoded activation backward pass : aka autodiff somehow
- More layers
- Type : Tensor?
- Object det augmentations

## Inspired by
- Karpathy and his awesomeness xD
- [micrograd](https://github.com/karpathy/micrograd)
- [teddykokker](https://github.com/teddykoker/tinyloader)
- ft. Every attempt of me trying this in other languages and failing miserably ):

## References
- The ones in Inspired by
- [training and tensor operations](https://github.com/kartik4949/deepops)
- [autograd](https://github.com/karpathy/micrograd)
- [pbar](https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console)
- [skalski](https://github.com/SkalskiP/ILearnDeepLearning.py/blob/master/01_mysteries_of_neural_networks/03_numpy_neural_net/Numpy%20deep%20neural%20network.ipynb)
- [softmax](https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python)
- [activations1](https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/)
- [albumentations](https://albumentations.ai/docs/api_reference/)
- [conv](https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381)
- [optim](https://github.com/ilguyi/optimizers.numpy/blob/master/02.stochastic.gradient.descent.ipynb)
