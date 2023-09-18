import os

import cv2 as cv
import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from readimgs import read_renoir


# Function to resize an image to a fixed size
def resize_image(image, target_size=(1024, 1024)):
    return cv.resize(image, target_size)

# Function to create patches from an image with a specified patch size
def create_patches(image, patch_size=(256, 256)):
    height, width = image.shape[:2]
    patches = []

    for y in range(0, height, patch_size[0]):
        for x in range(0, width, patch_size[1]):
            patch = image[y:y+patch_size[0], x:x+patch_size[1]]
            if patch.shape == patch_size:
                patches.append(patch)

    return patches


def patchify(target, label, patch_size=(256, 256)):
    print("Creating patches...")
    # Iterate through the images
    X, y = [], []
    for i in range(len(target)):
        # Resize the image to 1024x1024
        tg = resize_image(target[i])
        lb = resize_image(label[i])

        # Create patches from the resized image
        tg_patches = create_patches(tg)
        lb_patches = create_patches(lb)

        # Append the patches to the list
        X.extend(tg_patches)
        y.extend(lb_patches)

    # Convert the lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    print("Image resizing and patch creation complete.")
    return X, y


class Dataloder(tf.keras.utils.Sequence):    
    def __init__(self, X,y,batch_size=1, shuffle=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(X))

    def __getitem__(self, i):
        # collect batch data
        batch_x = self.X[i * self.batch_size : (i+1) * self.batch_size]
        batch_y = self.y[i * self.batch_size : (i+1) * self.batch_size]
        
        return tuple((batch_x,batch_y))
    
    def __len__(self):
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
            

# batch_size=32
# train_dataloader = Dataloder(X_train_patches,y_train_patches, batch_size, shuffle=True)
# test_dataloader = Dataloder(X_test_patches,y_test_patches,batch_size, shuffle=True) 

