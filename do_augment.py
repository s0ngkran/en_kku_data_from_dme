import cv2
import numpy as np
import json
import random

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, -angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def rotate_point(keypoint, angle):
    w, h = 360, 360
    center = (w / 2, h / 2)
    def rotate_point_(origin, p, degrees):
        angle = np.deg2rad(degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
        o = np.atleast_2d(origin)
        p = np.atleast_2d(p)
        return np.squeeze((np.matmul(R  ,(p.T-o.T)) + o.T).T)

    return [list(rotate_point_(center, p, angle)) for p in keypoint]

def aug_rotate(img, keypoint):
    angle = random.random()*10 - 5 # range [-5, 5]
    rotated_img = rotate(img, angle)

    rotated_keypoint = rotate_point(keypoint, angle)
    return rotated_img, rotated_keypoint

def point_scale(scale, gt_point, shift):
    shift_x, shift_y = shift
    ans = []
    for x,y in gt_point:
        new_x = shift_x + x*scale
        new_y = shift_y + y*scale 
        ans.append([int(new_x), int(new_y)])
    return ans

def aug_scale_and_shift(img, keypoint):
    scale = 1 - random.random()*0.08 # range [0.92, 1]
    img_size = img.shape
    dim = int(img_size[0]*scale), int(img_size[1]*scale)
    dimy, dimx = dim[0], dim[1]
    max_shift = img_size[0] - dim[0]

    new_img = img.copy()
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    x_shift = random.randint(0,max_shift)
    y_shift = random.randint(0,max_shift)
    new_img[y_shift:y_shift+dimy, x_shift:x_shift+dimx] = resized

    shift = (x_shift, y_shift)
    new_keypoint = point_scale(scale, keypoint, shift)
    return new_img, new_keypoint

def aug_flip(img, keypoint):

    img_size = img.shape
    new_img = cv2.flip(img, 1) # flip horizontal
    new_keypoint = []
    for x, y in keypoint:
        x = img_size[0] - x
        # y = old_value
        new_keypoint.append([x, y])
    return new_img, new_keypoint

def aug_noise(img, keypoint):
    mu, sigma = random.random()*4, random.random()*8 # mean and standard deviation
    noise = np.random.normal(mu, sigma, img.shape)
    new_img = np.add(img, noise)
    new_img[new_img>=255] = 255
    new_img[new_img<=0] = 0
    new_keypoint = keypoint
    return new_img, new_keypoint
