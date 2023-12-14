import setup_path
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import preprocess_input
import tensorflow as tf
import numpy as np
import cv2 as cv
import math

class BlurDetector:
    def preprocess_data(self, X):
        X = X/255
        return X

    # Function to create patches from an image with a specified patch size
    def split_image(self, image, patch_size=256):

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

        # Initialize an empty list to store the patches
        patches = []

        # Iterate through the image and extract patches
        for y in range(0, out_height, patch_size):
            for x in range(0, out_width, patch_size):
                patch = image[y:y+patch_size, x:x+patch_size]
                patches.append(patch)

        return np.array(patches)

    def detect_blur(self, image):
        model = tf.keras.applications.densenet.DenseNet121(
            include_top=False,
            weights=None,
            input_shape=(256, 256, 3)
        )

        top = GlobalAveragePooling2D()(model.output)
        top = Dense(1024, activation='relu')(top)
        top = Dense(1, activation='sigmoid')(top)
        model = Model(inputs=model.input, outputs=top)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # Load the model
        model.load_weights('model_zoo/blur_detect_checkpoint_path.h5')
        # Preprocess the image
        to_detect = self.split_image(image, 256)
        to_detect = self.preprocess_data(to_detect)
        # Detect if image has blur
        result = model.predict(to_detect)
        result = np.sum(np.round(result))/len(result)
        return result
