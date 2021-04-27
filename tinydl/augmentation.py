import numpy as np

def augment(X_images, image_shape, aug_list):
    """
    Args:
        X_images
        image_shape
        aug_list
    Performs a sequence of augmentations
    """
    num_images, xs, ys, nc = X_images.shape
    for augs in aug_list:
        X_images = augs(X_images)
    return X_images

def Normalize(X_images):
    """
    Args: 
        image array
    Normalizes the image
    """
    return (X_images - X_images.min()) / (X_images.max() - X_images.min())
