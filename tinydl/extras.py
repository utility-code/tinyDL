# LAYERS
def conv1d(image, kernel, pad = "same"):
    return np.convolve(image, kernel, mode= pad)

def conv2d(image, kernel, pad=2, stride=2):
    # cross correlation, flip horizontally then vertically
    kernel = np.flipud(np.fliplr(kernel))
    xkshape, ykshape = kernel.shape
    ximgshape, yimgshape = image.shape

    xout = int(((ximgshape - xkshape + 2 * pad)/stride)+1)
    yout = int(((yimgshape - ykshape + 2 * pad)/stride)+1)
    output = np.zeros((xout, yout))

    if pad != 0:
        imagepadded = np.zeros(
            (image.shape[0] + pad*2, image.shape[1] + pad*2))
        imagepadded[int(pad):int(-1*pad), int(pad):int(-1*pad)] = image
    else:
        imagepadded = image

    for y in range(image.shape[1]):
        if y > image.shape[1] - ykshape:
            break
        if y % stride == 0:
            for x in range(image.shape[0]):
                if x > image.shape[0] - xkshape:
                    break
                try:
                    if x % stride == 0:
                        output[x, y] = (
                            kernel*imagepadded[x:x+xkshape, y: y + ykshape]).sum()
                except:
                    break

    return {"value":output, "name":"conv2d"}


# ACTIVATIONS
def sigmoid(x):
    return {"value": 1/(1+np.exp(-x)), "name": "sigmoid"}

def relu(x,eps = 1e-14):
    return {"value": np.maximum(0, x)+eps, "name": "relu"}


def prelu(x, a):  # PRelu
    return {"value": np.maximum(0, x*a), "name": "prelu"}


def lrelu(x, alpha):  # leaky relu
    return {"value": np.maximum(alpha*x, x), "name": "lrelu"}


def softplus(x):
    return {"value": np.log(np.exp(x) + 1), "name": "softplus"}


def elu(x, a):
    return {"value": np.maximum(x, a*(np.exp(x)-1)), "name": "elu"}


def swish(x, beta):
    return {"value": x/(1+np.exp(-beta*x)), "name": "swish"}


def tanh(x):
    return {"value": (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)), "name": "tanh"}


def softmax(x):
    eX = np.exp(x - np.max(x))
    return {"value": eX/eX.sum(axis=0), "name": "softmax"}
