import random
import math
import torchvision.transforms.functional as F
import numpy as np
import cv2
import json
import torchvision.transforms as transforms
from PIL import Image

def DE_COLOR(img, color_params):
    """Randomly change the brightness, contrast and saturation of an image"""

    img = F.adjust_brightness(img, color_params['brightness_factor'])   
    img = F.adjust_contrast(img, color_params['contrast_factor'])  
    img = F.adjust_saturation(img, color_params['saturation_factor'])

    return img


def DE_HALO(img, halo_params, h=512, w=512):
    '''
    Defined to simulate a 'ringlike' halo noise in fundus image
    :param weight_r/weight_g/weight_b: Designed to simulate 3 kinds of halo noise shown in Kaggle dataset.
    :param center_a/center_b:          Position of each circle which is simulated the ringlike shape
    :param dia_a/dia_b:                Size of each circle which is simulated the ringlike noise
    :param weight_hal0:                Weight of added halo noise color
    :param sigma:                      Filter size for final Gaussian filter
    '''
    
    weight_r = [251/255,141/255,177/255]
    weight_g = [249/255,238/255,195/255]
    weight_b = [246/255,238/255,147/255]
    num = halo_params['num'] # num

    # a
    center_a = halo_params['center_a']
    dia_a = halo_params['dia_a']
    Y_a, X_a = np.ogrid[:h, :w]
    dist_from_center_a = np.sqrt((X_a - center_a[0]) ** 2 + (Y_a - center_a[1]) ** 2)
    circle_a = dist_from_center_a <= (int(dia_a / 2))
    mask_a = np.zeros((h, w))
    mask_a[circle_a] = np.mean(img)

    # b
    center_b = halo_params['center_b']
    dia_b = halo_params['dia_b']
    Y_b, X_b = np.ogrid[:h, :w]
    dist_from_center_b = np.sqrt((X_b - center_b[0]) ** 2 + (Y_b - center_b[1]) ** 2)
    circle_b = dist_from_center_b <= (int(dia_b / 2))
    mask_b = np.zeros((h, w))
    mask_b[circle_b] = np.mean(img)

    weight_hal0 = [0, 1, 1.5, 2, 2.5]
    delta_circle = np.abs(mask_a - mask_b) * weight_hal0[1]
    dia = max(center_a[0],h-center_a[0],center_a[1],h-center_a[1])*2
    gauss_rad = int(np.abs(dia-dia_a))
    sigma = 2/3*gauss_rad
    if(gauss_rad % 2) == 0:
        gauss_rad= gauss_rad+1
    delta_circle = cv2.GaussianBlur(delta_circle, (gauss_rad, gauss_rad), sigma)
    delta_circle = np.array([weight_r[num]*delta_circle,weight_g[num]*delta_circle,weight_b[num]*delta_circle])
    img = img + delta_circle
    img = np.maximum(img, 0)
    img = np.minimum(img, 1)

    return img


def DE_HOLE(img, region_mask, hole_params, h=512, w=512):
    '''

    :param diameter_circle:     The size of the simulated artifacts caused by non-uniform lighting
    :param center:              Position
    :param brightness_factor:   Weight utilized to adapt the value of generated non-uniform lighting artifacts.
    :param sigma:               Filter size for final Gaussian filter

    :return:
    '''
    # if radius is None: # use the smallest distance between the center and image walls
    # diameter_circle = random.randint(int(0.3*w), int(0.5 * w))
    #  define the center based on the position of disc/cup
    diameter_circle = hole_params['diameter_circle']
    center = hole_params['center']
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    circle = dist_from_center <= (int(diameter_circle/2))
    mask = np.zeros((h, w))
    mask[circle] = 1
    brightness_factor = hole_params['brightness_factor']
    mask = mask * brightness_factor

    rad_w = hole_params['rad_w']
    rad_h = hole_params['rad_h']
    sigma = hole_params['sigma']
    mask = cv2.GaussianBlur(mask, (rad_w, rad_h), sigma)
    mask = np.array([mask, mask, mask])
    img = img + mask
    img = np.maximum(img, 0)
    img = np.minimum(img, 1)

    return img

def DE_SPOT(img, spot_params, h=512, w=512):
    '''
    :param s_num:  The number of the generated artifacts spot on the fundus image
    :param radius: Define the size of each spot
    :param center: Position of each spot on the fundus image
    :param K:      Weight of original fundus image value
    :param beta:   Weight of generated artifacts(spots) mask value (The color is adapted based on the size(radius) of each spot)
    :param sigma:  Filter size for final Gaussian filter

    '''
    for spot_dict in range(spot_params):
        # if radius is None: # use the smallest distance between the center and image walls
        # radius = min(center[0], center[1], w-center[0], h-center[1])
        radius = spot_dict['radius']

        # if center is None: # in the middle of the image
        center  = spot_dict['center']
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        circle = dist_from_center <= (int(radius/2))

        k =(14/25) +(1.0-radius/25)
        beta = 0.5 + (1.5 - 0.5) * (radius/25)
        A = k *np.ones((3,1))
        d =0.3 *(radius/25)
        t = math.exp(-beta * d)

        mask = np.zeros((h, w))
        mask[circle] = np.multiply(A[0],(1-t))

        sigma = spot_dict['sigma']
        rad_w = spot_dict['rad_w']
        rad_h = spot_dict['rad_h']

        mask = cv2.GaussianBlur(mask, (rad_w, rad_h), sigma)
        mask = np.array([mask,mask,mask])
        img = img + mask
        img = np.maximum(img,0)
        img = np.minimum(img,1)

    return img

def DE_BLUR(img, blur_params, h=512, w=512):
    '''

    :param sigma: Filter size for Gaussian filter

    '''
    img = (np.transpose(img, (1, 2, 0)))
    sigma = blur_params['sigma']
    rad_w = blur_params['rad_w']
    rad_h = blur_params['rad_h']
    img = cv2.GaussianBlur(img, (rad_w,rad_h), sigma)
    img = (np.transpose(img, (2, 0, 1)))

    img = np.maximum(img, 0)
    img= np.minimum(img, 1)


    return img
