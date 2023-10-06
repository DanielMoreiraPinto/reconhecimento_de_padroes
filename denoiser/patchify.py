import numpy as np


def split_image(image, patch_size=256):
    height, width, _ = image.shape
    patches = []

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            if patch.shape[:2] != (patch_size, patch_size):
                # Pad the patch with zeros if it's smaller than the patch_size
                pad_height = patch_size - patch.shape[0]
                pad_width = patch_size - patch.shape[1]
                patch = np.pad(patch, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
            patches.append(patch)

    return patches

def reconstruct_image(patches, image_shape):
    height, width, _ = image_shape
    patch_size = patches[0].shape[0]

    reconstructed_image = np.zeros(image_shape, dtype=np.uint8)

    patch_idx = 0
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = patches[patch_idx]
            reconstructed_image[y:y+patch_size, x:x+patch_size] = patch
            patch_idx += 1

    return reconstructed_image

