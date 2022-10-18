import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def p_within_bounding_box(patch_box, bound_box):
    """
    Get percent of patch box within bounding box geven points
    """
    overlap_width = max(min(patch_box[0], bound_box[0]) - max(patch_box[1], bound_box[1]), 0)
    overlap_height = max(min(patch_box[2], bound_box[2]) - max(patch_box[3], bound_box[3]), 0)
    return (overlap_width * overlap_height) / ((patch_box[0] - patch_box[1]) * (patch_box[2] - patch_box[3]))

def get_patch_data(imgs, img_bboxes, patch_size):
    """
    Get patch data given a patch size
    """
    img_patches = []
    p_within_bounds = []

    # Get list of patches for each image
    for img, bboxes in zip(imgs, img_bboxes):
        for i in range(0, img.shape[0] // patch_size):
            for j in range(0, img.shape[1] // patch_size):
                y_max = (i * patch_size) + patch_size
                y_min = i * patch_size
                x_max = (j * patch_size) + patch_size
                x_min = j * patch_size
                
                # Get proportion of patch in each bounding box
                ps_within_bounds = []
                for bbox in bboxes:
                    ps_within_bounds.append(p_within_bounding_box((x_max, x_min, y_max, y_min), bbox))
                
                img_patches.append(img[y_min:y_max,x_min:x_max,:])
                p_within_bounds.append(max(ps_within_bounds))

    return img_patches, p_within_bounds

# Data transformations
def prepare_data(img_patches, p_in_bounds, binary=False):
    return np.moveaxis(np.array(img_patches), 3, 1), [int(p > 0) if binary else p for p in p_in_bounds]


def plot_bounded_img(img, bboxes):
    fig, ax = plt.subplots()
    plt.imshow(img)
    for box in bboxes:
        ax.add_patch(patches.Rectangle((box[1], box[3]), box[0] - box[1], box[2] - box[3], linewidth=1, edgecolor='r', facecolor='none'))
    plt.show()