import os
import json
import numpy as np

import matplotlib.pyplot as plt
import cv2
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

import torch

def cluster_img(img, k, c=3, n=10, t=1., return_binary=False):
    """ Apply kmeans clusttering to image """
    img = img.copy()
    img = (img - np.min(img)) / np.max(img)
    img *= 255.

    vectorized = np.float32(img.reshape((-1, c)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, n, t)

    attempts=10
    ret, label, center = cv2.kmeans(
        vectorized, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )

    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    result = result_image / 255.

    if return_binary:
        return (result > (1 / (k + 1))).astype(np.float32)
    return result

# Filter out the small negative regions based on their size
def fill_gaps(img, stride, size_thres, border_size=20):
    """ Apply flood fill to fill gaps """
    img = img.copy()

    # Remove border from image to ensure flooding coverage
    img[:border_size, :] = 0
    img[-border_size:, :] = 0
    img[:,:border_size] = 0
    img[:,-border_size:] = 0

    # Flood outer part of image
    flooded_img = cv2.floodFill(np.float32(img.copy()), None, (0, 0), 1)[1]

    return img + filter_negative_regions(img, 1 - flooded_img, stride, size_thres)

# Filter the negative regions based on its size
def filter_negative_regions(img, negative_regions, stride, size_thres):
    img = img.copy()
    negative_regions = negative_regions.copy()
    large_negative_regions = np.zeros((img.shape[0], img.shape[1]))
    centroids = []
    for i in range(0, img.shape[0], stride):
        for j in range(0, img.shape[1], stride):
            if negative_regions[i, j] == 1:
                selected_region = cv2.floodFill(np.float32(negative_regions.copy()), None, (j, i), 0.5)[1] == 0.5
                if float(selected_region.sum()) > size_thres * img.sum():
                    large_negative_regions += selected_region
                    negative_regions = np.float32(negative_regions > 0) - np.float32(selected_region)
    return negative_regions > 0

def cluster_fill(act_map, stride=10, size_thres=0.01):
    return fill_gaps(cluster_img(act_map, 2) > 0, stride, size_thres)
    
def erode_segmentation(seg_region, k, n):
    return cv2.erode(np.float32(seg_region), kernel=np.ones((k, k), np.uint8), iterations=n)
    
# Create image bounding box masks
def create_img_bbox_mask(img, bboxes):
    img_bbox_mask = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img_bbox_mask.shape[0]):
        for j in range(img_bbox_mask.shape[1]):
            for bbox in bboxes:
                if j <= bbox[0] and j >= bbox[1] and i <= bbox[2] and i >= bbox[3]:
                    img_bbox_mask[i][j] = 1

    return img_bbox_mask

# Create cropped noisy and clean data
def get_center_cropped_image(img, shape):
    x_start = (img.shape[0] - shape[0]) // 2
    x_end = (img.shape[0] - shape[0]) // 2 + shape[0]
    y_start = (img.shape[1] - shape[1]) // 2
    y_end = (img.shape[1] - shape[1]) // 2 + shape[1]
    
    cropped_img = img[x_start:x_end, y_start:y_end]

    if cropped_img.shape != shape:
        print(img.shape)
        print(shape)
        print(x_start, x_end, y_start, y_end)
        plt.imshow(cropped_img)
        plt.show()

    return cropped_img

def get_noise(shape, noise_scale_factor):
    noise = np.random.normal(0, 1, [s // noise_scale_factor for s in shape])
    noise = 1 / (1 + np.exp(noise))
    noise = cv2.resize(noise, tuple(reversed(shape)))
    return noise

def get_saliency_map(model, layer_i, img, s, gpu=False):
    img_tensor = torch.unsqueeze(img, 0)
    if gpu: img_tensor = img_tensor.cuda()
    feature_map = model.saliency_map(img_tensor, layer_i).detach()
    if gpu: feature_map = feature_map.cpu()
    saliency_map = cv2.resize(np.mean(feature_map.numpy()[0], axis=0), tuple(reversed(img.shape[1:])))
    return (2 * (1 / (1 + np.exp(-saliency_map / s))) - 1)

def get_predicted_region(model, img, s=12, patch_size=64, gpu=False):
    img_tensor = torch.unsqueeze(img, 0)
    if gpu: img_tensor = img_tensor.cuda()

    predicted_region = torch.zeros((img.shape[1], img.shape[2]))

    for i in range(0, img.shape[1], patch_size // 4):
        for j in range(0, img.shape[2], patch_size // 4):
            x = torch.unsqueeze(img[:,i:i+patch_size,j:j+patch_size], dim=0)
            if x.shape[-1] == x.shape[-2] and x.shape[-1] == patch_size:
                predicted_region[i:i+patch_size,j:j+patch_size] += model(x)[0].item()

    return (2 * (1 / (1 + np.exp(-predicted_region / s))) - 1)

