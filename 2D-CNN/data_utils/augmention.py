#  -*- coding: utf-8 -*-
'''
augmentation methods
'''
import numpy as np
import cv2
import imutils
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt



# flip
def flip_img(img, flip_type):
    if flip_type == 'V':
        flip_img = cv2.flip(img, 0)
    elif flip_type == 'H':
        flip_img = cv2.flip(img, 1)
    return flip_img






#rescale
def rescale_transform(img, scale):
    rescale_img = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return rescale_img





# Mosaic
def mosaic_transform(img, scale):
    shape_size = img.shape[:2]
    down_scale = 1.0/scale
    mosaic_img_temp = cv2.resize(img, (0,0), fx=down_scale, fy=down_scale, interpolation=cv2.INTER_LINEAR)
    mosaic_img = cv2.resize(mosaic_img_temp, (shape_size[1], shape_size[0]), interpolation=cv2.INTER_NEAREST)
    return mosaic_img





# rotation
def rotation_transform(img, angle, borderMode=cv2.BORDER_CONSTANT):
    h, w = img.shape[:2]
    rotate_center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
    new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated_img = cv2.warpAffine(img, M, (new_w, new_h),flags=cv2.INTER_LINEAR,
                                borderMode=borderMode,
                                borderValue=(255, 255, 255,))
    return rotated_img






# Brightness
def brightness_shift(img, shift_max_value):
    img = img.astype(np.float64) / 255.0
    img += np.random.uniform(-shift_max_value, shift_max_value)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img


def brightness_scaling(src, max_value):
    img = src.copy()
    img = img.astype(np.float64) / 255.0
    img *= np.random.uniform(1 - max_value, 1 + max_value)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img



# Gamma correction（HSV中对V进行gamma correction）
def gamma_correction(src, gamma):
    img = src.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    channels = cv2.split(hsv)
    temp = channels[2].astype(np.float64) / 255.0
    temp = np.power(temp, gamma)
    channels_temp = (np.clip(temp, 0, 1) * 255).astype(np.uint8)
    channels = tuple((channels[0], channels[1], channels_temp))
    hsv = (cv2.merge(channels, hsv)).astype(np.uint8)
    cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB, img)
    return img


# (Adaptive) Histogram Equation, Histogram Equalization
def hist_equation_color(src):
    img = src.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    channels = cv2.split(hsv)
    cv2.equalizeHist(channels[2], channels[2])
    cv2.merge(channels, hsv)
    cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB, img)
    return img



# https://docs.opencv.org/4.1.0/d5/daf/tutorial_py_histogram_equalization.html
def adapt_hist_equation_color(src):
    img = src.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    channels = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[2], channels[2])
    cv2.merge(channels, hsv)
    cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB, img)
    return img





def draw_grid(img, grid_size):
    # Draw grid lines
    for i in range(0, img.shape[1], grid_size):
        cv2.line(img, (i, 0), (i, img.shape[0]), color=(255,))
    for j in range(0, img.shape[0], grid_size):
        cv2.line(img, (0, j), (img.shape[1], j), color=(255,))
    plt.imshow(img, interpolation='nearest')
    plt.show()




# shearing
def horizontal_shear(image, scale, is_show, borderMode=cv2.BORDER_CONSTANT):
    if is_show:
        draw_grid(image, 50)
    height, width = image.shape[:2]
    dx = int(scale * width)
    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ], np.float32)
    box1 = np.array([[+dx, 0], [width + dx, 0], [width - dx, height], [-dx, height], ], np.float32)
    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)
    aug_image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=borderMode, borderValue=(255, 255, 255,))
    return aug_image






# Elastic Transform
def elastic_transform_kerl(image, beta, random_state=None, borderMode=cv2.BORDER_CONSTANT):
    """Elastic deformation of images"""
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    alpha = image.shape[1] * 2
    sigma = image.shape[1] * beta
    alpha_affine = image.shape[1] * beta * 0.01

    # Random affine
    square_size = min(shape_size) // 3
    center_square = np.float32(shape_size) // 2
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=borderMode, borderValue=(255, 255, 255,))

    # random displacement fields
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    #generate meshgrid
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)





def elastic_transform(im, alpha, is_show, random_state=None, borderMode=cv2.BORDER_CONSTANT):
    # Add grid lines for show
    if is_show:
        draw_grid(im, 50)
    im_t = elastic_transform_kerl(im, alpha, random_state, borderMode)
    return im_t






# Add noise
def add_noisy(image, noise_typ):
    image = image.astype(np.float64) / 255
    row, col, ch = image.shape
    if noise_typ == "gauss":
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        result = image + gauss
        result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        return result

    elif noise_typ == "s_p":
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        result = out
        result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        return result

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        result = np.random.poisson(image * vals) / float(vals)
        result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        return result

    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        result = image + image * gauss
        result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        return result

