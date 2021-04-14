# TinyDL

- Tiny Deep learning library 
- [DOCUMENTATION](https://subhadityamukherjee.github.io/tinyDL/)
- Check the demo.py or the demo notebook!!! (same content)
- Have a look at what works so far :)
- New features will be added as time goes
- Only needs numpy and pycuda for the library

## Requirements
- numpy
- matplotlib
- pycuda  (for GPU)
- pandas (if you are using the helpers and need to read a table)

## How to run
- Install the requirements
- Configure parameters in config.py
- python demo.py

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

## What are the .sh files
- I am lazy. So I write code when I have to do the same things again and again.
### pusher.sh
- Formats all the code using "Black" formatter
- Creates documentation from docstrings in the code using pandoc
- Fixes the documentation paths for working with Github Pages
- Takes the demo.py and pops it into a nice notebook for anyone to run and use
- If an argument is given, it git commits with the message and pushes it to the repository

## References
- [training and tensor operations](https://github.com/kartik4949/deepops)
- [autograd](https://github.com/karpathy/micrograd)
- [pbar](https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console)
- [skalski](https://github.com/SkalskiP/ILearnDeepLearning.py/blob/master/01_mysteries_of_neural_networks/03_numpy_neural_net/Numpy%20deep%20neural%20network.ipynb)
- [softmax](https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python)
- [activations1](https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/)
- [albumentations](https://albumentations.ai/docs/api_reference/)
- [conv](https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381)
- [optim](https://github.com/ilguyi/optimizers.numpy/blob/master/02.stochastic.gradient.descent.ipynb)
- Countless stackoverflow searches (hehe)
