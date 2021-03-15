# TinyDL(WIP)

- Tiny Deep learning library
- Super WIP
- New features will be added as time goes

## How to run
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

## Whats there so far
### Basic dataloader
- Supports multiprocessing
- Accepts int, float, numpy arrays
- Note : No tensors yet
### Layers
- sigmoid
- relu
### Progress bar
- A progress bar generator xD (I was going to use tqdm. But well.)
- If you have a small screen/terminal window, reduce the length parameter

## Requirements
- numpy

## Lazy list on what to add
- (Just adding stuff here when I notice something. Not exhaustive and prone to huge changes)
- Avoid using hardcoded activation backward pass : aka autodiff somehow
- All the activation functions
- More layers
- Type : Tensor?
- GPU (Super long run... Probably not lol)
