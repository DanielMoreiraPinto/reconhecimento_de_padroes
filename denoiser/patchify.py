import math

import numpy as np
import cv2 as cv


def split_image(image, target=None, patch_size=256, patch_ratio=0, random_seed=None):
    height, width, _ = image.shape

    # Calculate the number of rows and columns needed for patches
    num_rows = math.ceil(height / patch_size)
    num_cols = math.ceil(width / patch_size)
    
    out_height = patch_size * num_rows
    out_width = patch_size * num_cols

    # Calculate the amount of padding needed
    pad_height = out_height - height
    pad_width = out_width - width

    # Pad the image with zeros if necessary
    if pad_height > 0 or pad_width > 0:
        image = cv.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv.BORDER_CONSTANT, value=0)
        if target is not None:
            target = cv.copyMakeBorder(target, 0, pad_height, 0, pad_width, cv.BORDER_CONSTANT, value=0)

    # Initialize an empty list to store the patches
    patches = []
    target_patches = []

    # Iterate through the image and extract patches
    for y in range(0, out_height, patch_size):
        for x in range(0, out_width, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            if target is not None:
                target_patch = target[y:y+patch_size, x:x+patch_size]
                target_patches.append(target_patch)
            patches.append(patch)

    if patch_ratio > 0:
        if random_seed is not None:
            np.random.seed(random_seed)
        num_patches = len(patches)
        num_patches_to_keep = math.ceil(num_patches * patch_ratio)
        patch_indices = np.random.choice(num_patches, num_patches_to_keep, replace=False)
        patches = [patches[i] for i in patch_indices]
        if target is not None:
            target_patches = [target_patches[i] for i in patch_indices]

    if target is not None:
        return np.array(patches), np.array(target_patches)
    return np.array(patches)

def reconstruct_image(patches, image_shape):
    # Create an empty array for the reconstructed image
    reconstructed_image = np.zeros(image_shape, dtype=patches.dtype)
    patches_inside_img = image_shape[0] // patches.shape[1]
    
    # Initialize variables to keep track of current position
    current_row, current_col = 0, 0
    
    # Iterate through the patches
    for patch in patches:
        # Calculate the maximum row and column indices for this patch
        max_row = current_row + patch.shape[0]
        max_col = current_col + patch.shape[1]
        
        # If the patch goes beyond the reconstructed image, 
        # cut it off to fit
        if max_row > image_shape[0]:
            patch = patch[0:image_shape[0]-current_row, :, :]
            max_row = image_shape[0]
        if max_col > image_shape[1]:
            patch = patch[:, 0:image_shape[1]-current_col, :]
            max_col = image_shape[1]
            
        # Add the patch to the reconstructed image
        reconstructed_image[current_row:max_row, current_col:max_col, :] = patch
        
        # Update the current row and col positions
        current_col += patch.shape[1]
        if current_col >= image_shape[1]:
            current_row += patch.shape[0]
            current_col = 0
    
    return reconstructed_image

