# TinyDL(WIP)

- Tiny Deep learning library
- Super WIP
- New features will be added as time goes
- Only needs numpy for the library

## Requirements
- numpy

## How to run
- Configure parameters in config.py
- python main.py

## Why?
- Pytorch is too complicated to learn from. (Please. I tried. There are a million folders. My brain >.<)
- This does not intend to be Pytorch. Just to understand what goes into it
- An attempt at recreating most of the essential components from scratch
- Eventual blogs on it as well

## Inspired by
- Karpathy and his awesomeness xD
- [micrograd](https://github.com/karpathy/micrograd)
- [teddykokker](https://github.com/teddykoker/tinyloader)
- ft. Every attempt of me trying this in other languages and failing miserably ):

## References
- The ones in Inspired by
- [pbar](https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console)
- [skalski](https://github.com/SkalskiP/ILearnDeepLearning.py/blob/master/01_mysteries_of_neural_networks/03_numpy_neural_net/Numpy%20deep%20neural%20network.ipynb)
- [softmax](https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python)
- [activations1](https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/)

## Whats there so far
### Unified config file
- Okay it took me 2 minutes to add but I am proud of it
### Complete training
- Forward/backprop
- Loss
- Gradient Descent
### Basic dataloader
- Supports multiprocessing
- Accepts int, float, numpy arrays
- Note : No tensors yet
### Layers/Activations
- linear
- sigmoid
- relu
- prelu
- leaky relu
- softmax
- softplus
- elu
- swish
- tanh
### Progress bar
- A progress bar generator xD (I was going to use tqdm. But well.)
- If you have a small screen/terminal window, reduce the length parameter
### Loss functions
- MSE

## Lazy list on what to add
- (Just adding stuff here when I notice something. Not exhaustive and prone to huge changes)
- Avoid using hardcoded activation backward pass : aka autodiff somehow
- More layers
- Type : Tensor?
- GPU (Super long run... Probably not lol)
- Conv xD
