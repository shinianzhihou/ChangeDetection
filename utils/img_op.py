import cv2
import numpy as np


def to_grad(im=None):
    '''Calculate the gradient of images by Sobel operator.

    Args:
        im:
            the image to be calculate

    Return:
        return a list consist of gradient, degree, and the splicing of both

        example:
            [gradient ,degree ,np.concatenate(gradient,degree)]

    '''
    m = 1e-5  # small constant used to avoid dividing 0
    im = im[:, :, 0:1]
    H, W, C = im.shape
    filterx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ])
    filtery = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ])
    gradx = cv2.filter2D(im, -1, filterx).reshape((H, W, -1))
    grady = cv2.filter2D(im, -1, filtery).reshape((H, W, -1))
    grad = np.sqrt(np.square(gradx) + np.square(grady)).reshape((H, W, -1))
    degree = np.arccos(gradx / (grad + m)).reshape((H, W, -1))
    return grad, degree, np.concatenate((grad, degree), axis=2)
