import tensorflow as tf
import cv2
import numpy as np


def load_model(model_path):
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded.")
    return model

def resize_image(image, target_size=(1024, 1024)):
    return cv2.resize(image, target_size)

def create_patches(image, patch_size=(256, 256)):
    height, width = image.shape[:2]
    patches = []

    for y in range(0, height, patch_size[0]):
        for x in range(0, width, patch_size[1]):
            patch = image[y:y+patch_size[0], x:x+patch_size[1]]
            patches.append(patch)

    return patches

def read_image(image_path):
    image = cv2.imread(image_path)
    return image

def join_patches(patches, image_size=(1024, 1024)):
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    patch_size = (256, 256)
    patches_per_row = image_size[0] // patch_size[0]
    patches_per_col = image_size[1] // patch_size[1]
    for i, patch in enumerate(patches):
        row = i // patches_per_row
        col = i % patches_per_row
        start_row = row * patch_size[0]
        end_row = start_row + patch_size[0]
        start_col = col * patch_size[1]
        end_col = start_col + patch_size[1] 
        image[start_row:end_row, start_col:end_col] = patch

    return image

def unpatchify(patches, image_size=(1024, 1024)):
    print("Unpatchifying images...")
    images = []
    patches_by_img = int((image_size[0] /256) ** 2)
    for i in range(0, len(patches), patches_by_img):
        image = join_patches(patches[i:i+patches_by_img])
        images.append(image)
    
    images = np.array(images)
    print("Unpatchifying complete.")
    return images


def denoise(image):
    # Load the model
    model = load_model('C:\\Users\\joao_\\Documents\\Trabalho-ReconhecimentoPadroes\\models\\simple_autoencoder.h5' )
    # Preprocess the image
    image = resize_image(image)
    image = create_patches(image)
    image = np.array(image)
    image = image / 255.0
    # Denoise the image
    image = model.predict(image)
    image = image * 255.0
    image = image.astype(np.uint8)
    image = unpatchify(image)
    return image[0]