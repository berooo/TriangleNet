"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from PIL import Image

import cv2
import numpy as np
import os
import scipy.misc as misc

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(images,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = images[0].shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)

        new_images = ()
        for img in images:
            new_images = new_images + (cv2.warpPerspective(img, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode, borderValue=(0, 0, 0)),)
        return new_images

    return images

def randomHorizontalFlip(images, u=0.5):
    if np.random.random() < u:
        new_images = ()
        for img in images:
            new_images = new_images + (cv2.flip(img, 1),)
        return new_images

    return images

def randomVerticleFlip(images, u=0.5):
    if np.random.random() < u:
        new_images = ()
        for img in images:
            new_images = new_images + (cv2.flip(img, 0),)
        return new_images

    return images

def randomRotate90(images, u=0.5):
    if np.random.random() < u:
        new_images = ()
        for img in images:
            new_images = new_images + (np.rot90(img),)
        return new_images

    return images


def default_loader(img_path, content_path, contour_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (448, 448))
    content = cv2.imread(content_path, cv2.IMREAD_GRAYSCALE)
    content = cv2.resize(content, (448, 448))
    contour = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
    contour = cv2.resize(contour, (448, 448))

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, content, contour = randomShiftScaleRotate((img, content, contour),
                                                    shift_limit=(-0.1, 0.1),
                                                    scale_limit=(-0.1, 0.1),
                                                    aspect_limit=(-0.1, 0.1),
                                                    rotate_limit=(-0, 0))
    img, content, contour = randomHorizontalFlip((img, content, contour))
    img, content, contour = randomVerticleFlip((img, content, contour))
    img, content, contour = randomRotate90((img, content, contour))

    mask = np.concatenate([content, contour], axis = 1)
    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    return img, mask

def default_content_loader(img_path, content_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (448, 448))
    content = cv2.imread(content_path, cv2.IMREAD_GRAYSCALE)
    content = cv2.resize(content, (448, 448))

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, content = randomShiftScaleRotate((img, content),
                                           shift_limit=(-0.1, 0.1),
                                           scale_limit=(-0.1, 0.1),
                                           aspect_limit=(-0.1, 0.1),
                                           rotate_limit=(-0, 0))
    img, content = randomHorizontalFlip((img, content))
    img, content = randomVerticleFlip((img, content))
    img, content = randomRotate90((img, content))

    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    return img, mask


def read_content_contour_datasets(root_path, mode='train'):
    images = []
    contour_masks = []
    content_masks = []

    image_root = os.path.join(root_path, 'train', 'images')
    gt_root = os.path.join(root_path, 'train', 'masks')

    for image_name in os.listdir(image_root):
        if image_name[:4] == 'TCGA':
            image_path = os.path.join(image_root, image_name)
            contour_path = os.path.join(gt_root, image_name[:23] + '_contour' + image_name[23:])
            content_path = os.path.join(gt_root, image_name[:23] + '_content' + image_name[23:])
            if os.path.exists(image_path) and os.path.exists(contour_path) and os.path.exists(content_path):
                images.append(image_path)
                contour_masks.append(contour_path)
                content_masks.append(content_path)

    return images, content_masks, contour_masks

def read_content_datasets(root_path, mode='train'):
    images = []
    content_masks = []

    image_root = os.path.join(root_path, 'train', 'images')
    gt_root = os.path.join(root_path, 'train', 'masks')

    for image_name in os.listdir(image_root):
        if image_name[:4] == 'TCGA':
            image_path = os.path.join(image_root, image_name)
            content_path = os.path.join(gt_root, image_name[:23] + '_content' + image_name[23:])
            if os.path.exists(image_path) and os.path.exists(content_path):
                images.append(image_path)
                content_masks.append(content_path)

    return images, content_masks


class ImageFolder(data.Dataset):

    def __init__(self, root_path, datasets='content_contour',  mode='train'):
        self.root = root_path
        self.mode = mode
        self.dataset = datasets
        assert self.dataset in ['content_contour', 'content'], \
            "the dataset should be in 'content_contour', 'content'"
        if self.dataset == 'content_contour':
            self.images, self.content_masks, self.contour_masks = read_content_contour_datasets(self.root, self.mode)
        elif self.dataset == 'content':
            self.images, self.content_masks = read_content_datasets(self.root, self.mode)
        else:
            print('Default dataset is content_contour')
            self.images, self.content_masks, self.contour_masks = read_content_contour_datasets(self.root, self.mode)

    def __getitem__(self, index):
        img = []
        mask = []
        if self.dataset == 'content_contour':
            img, mask = default_loader(self.images[index], self.content_masks[index], self.contour_masks[index])
        elif self.dataset == 'content':
            img, mask = default_content_loader(self.images[index], self.content_masks[index])
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        assert len(self.images) == len(self.content_masks), 'The number of images must be equal to labels'
        return len(self.images)