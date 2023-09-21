import sys
sys.path.append('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes')

import os

import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from denoiser.readimgs_renoir import read_renoir
from denoiser.networks import simple_autoencoder, cbd_net, rid_net, dn_cnn

np.random.seed(42)


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
        tg = create_patches(tg)
        lb = create_patches(lb)

        # Append the patches to the list
        X.extend(tg)
        y.extend(lb)

    # Convert the lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    print("Image resizing and patch creation complete.")
    return X, y

def join_patches(patches, image_size=(1024, 1024)):
    # Create an empty image
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    # Get the patch size
    patch_size = (256, 256)

    # Get the number of patches per row and column
    patches_per_row = image_size[0] // patch_size[0]
    patches_per_col = image_size[1] // patch_size[1]

    # Iterate through the patches
    for i, patch in enumerate(patches):
        # Calculate the row and column of the patch
        row = i // patches_per_row
        col = i % patches_per_row

        # Calculate the start and end row of the patch
        start_row = row * patch_size[0]
        end_row = start_row + patch_size[0]

        # Calculate the start and end column of the patch
        start_col = col * patch_size[1]
        end_col = start_col + patch_size[1]

        # Insert the patch into the image
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

def preprocess_data(X, y, shuffle):
    # Shuffle the data if specified
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

    return X, y

def train_model(X_train, y_train, X_val, y_val, model_type='simple_autoencoder', 
                model_path='data\\denoiser.h5', epochs=10, batch_size=32):
    train_data, train_label = preprocess_data(X_train, y_train, shuffle=True)
    val_data, val_label = preprocess_data(X_val, y_val, shuffle=True)

    # Create the model
    model = select_model(model_type)

    # Create validation loss checkpoint
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    print("Training model...")
    model.fit(train_data, train_label, batch_size=batch_size, epochs=epochs, validation_data=(val_data, val_label), callbacks=[checkpoint])
    print("Training complete.")

    return model

def select_model(model_type):
    if model_type == 'simple_autoencoder':
        model = simple_autoencoder()
    elif model_type == 'cbd_net':
        model = cbd_net()
    elif model_type == 'rid_net':
        model = rid_net()
    elif model_type == 'dn_cnn':
        model = dn_cnn()
    else:
        raise ValueError("Invalid model type. Valid model types are 'simple_autoencoder', 'cbd_net' and 'rid_net'.")
    return model

def load_model(model_path):
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded.")
    return model

def load_model_weights(model_path, model_type='simple_autoencoder'):
    print("Loading model...")
    model = select_model(model_type)
    model.load_weights(model_path)
    print("Model loaded.")
    return model

def test_model(model, X_test, y_test):
        # Predict with the model all images in the testing set
    print("Testing model...")
    y_pred = model.predict(X_test)
    X_test = X_test * 255.0
    X_test = X_test.astype(np.uint8)
    y_pred = y_pred * 255.0
    y_pred = y_pred.astype(np.uint8)
    y_test = y_test * 255.0
    y_test = y_test.astype(np.uint8)

    # Calculate the PSNR and SSIM for each image
    psnr_noisy_list = []
    ssim_noisy_list = []
    psnr_list = []
    ssim_list = []
    for i in range(len(y_test)):
        psnr_list.append(psnr(y_test[i], y_pred[i]))
        ssim_list.append(ssim(y_test[i], y_pred[i], channel_axis=-1))
        psnr_noisy_list.append(psnr(y_test[i], X_test[i]))
        ssim_noisy_list.append(ssim(y_test[i], X_test[i], channel_axis=-1))

    # Calculate the average PSNR and SSIM
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_psnr_noisy = np.mean(psnr_noisy_list)
    avg_ssim_noisy = np.mean(ssim_noisy_list)

    print("Testing complete.")
    print(f"Average PSNR: {avg_psnr}")
    print(f"Average SSIM: {avg_ssim}")
    print(f"Average PSNR (noisy): {avg_psnr_noisy}")
    print(f"Average SSIM (noisy): {avg_ssim_noisy}")

    metrics_by_image = [psnr_list, ssim_list, psnr_noisy_list, ssim_noisy_list]
    avg_metrics = [avg_psnr, avg_ssim, avg_psnr_noisy, avg_ssim_noisy]

    return y_pred, y_test, metrics_by_image, avg_metrics


RENOIR_DATASET_PATHS = ['D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\Mi3_Aligned',
                        'D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\S90_Aligned',
                        'D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\T3i_Aligned']
TEST_SAVE_PATH = 'D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\test'
TEST_SAMPLES_PATH = 'D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\denoiser\data\\test_sample'
MODEL_TYPE = 'simple_autoencoder'
# MODEL_TYPE = 'cbd_net'
# MODEL_TYPE = 'rid_net'
# MODEL_TYPE = 'dn_cnn'
# MODEL_PATH = f'D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\denoiser\\data\\models\\{MODEL_TYPE}.h5'
MODEL_PATH = f'D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\denoiser\\data\\models\\v2\\{MODEL_TYPE}.h5'
EPOCHS = 50
BATCH_SIZE = 4

def training(ckpt_path = None):
    # Read the images from the dataset
    X, y = read_renoir(RENOIR_DATASET_PATHS, num_images=0)

    # Divide the dataset into training, validation and testing sets
    # 80% training, 10% validation, 10% testing
    X_train, y_train = X[:int(len(X)*0.8)], y[:int(len(y)*0.8)]
    X_val, y_val = X[int(len(X)*0.8):int(len(X)*0.9)], y[int(len(y)*0.8):int(len(y)*0.9)]
    X_test, y_test = X[int(len(X)*0.9):], y[int(len(y)*0.9):]

    # Create patches from the images
    X_train, y_train = patchify(X_train, y_train)
    X_val, y_val = patchify(X_val, y_val)
    X_test, y_test = patchify(X_test, y_test)

    # Save the testing set
    if not os.path.exists(TEST_SAVE_PATH):
        os.makedirs(TEST_SAVE_PATH)
        np.save(os.path.join(TEST_SAVE_PATH, 'x_test.npy'), X_test)
        np.save(os.path.join(TEST_SAVE_PATH, 'y_test.npy'), y_test)
        print("Testing set saved.")
    else:
        print("Testing set already exists.")
    
    # Normalizing data
    X_train = X_train / 255.0
    y_train = y_train / 255.0
    X_val = X_val / 255.0
    y_val = y_val / 255.0
    X_test = X_test / 255.0
    y_test = y_test / 255.0
    
    # Train the model
    if ckpt_path is not None:
        # Load the model
        model = load_model(ckpt_path)
        # Test the model
        test_model(model, X_test, y_test)
    else:
        model = train_model(X_train, y_train, X_val, y_val, model_type=MODEL_TYPE, model_path=MODEL_PATH, epochs=EPOCHS, batch_size=BATCH_SIZE)
        # Test the last model
        test_model(model, X_test, y_test)
        # Test the best model
        model = load_model(MODEL_PATH)
        test_model(model, X_test, y_test)

def test_denoising(test_path, model_path, save_path):
    print("Loading testing set...")
    X_test = np.load(os.path.join(test_path, 'x_test.npy'))
    y_test = np.load(os.path.join(test_path, 'y_test.npy'))
    print("Testing set loaded.")
    X_test = X_test / 255.0
    y_test = y_test / 255.0

    print("Loading model...")
    model = load_model(model_path)
    print("Model loaded.")
    y_pred, y_test, metrics_by_image, avg_metrics = test_model(model, X_test, y_test)
    y_test = unpatchify(y_test)
    y_pred = unpatchify(y_pred)
    X_test = X_test * 255.0
    X_test = X_test.astype(np.uint8)
    X_test = unpatchify(X_test)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(len(y_test)):
        cv.imwrite(os.path.join(save_path, f'{i}_noise.png'), X_test[i])
        cv.imwrite(os.path.join(save_path, f'{i}_true.png'), y_test[i])
        cv.imwrite(os.path.join(save_path, f'{i}_pred.png'), y_pred[i])
        if i == 19:
            break
    print("Testing complete.")


if __name__ == "__main__":
    # training()
    # training(ckpt_path=MODEL_PATH)
    # test_denoising(TEST_SAVE_PATH, MODEL_PATH, TEST_SAMPLES_PATH)
    pass


def denoise(image):
    # Load the model
    model = load_model(MODEL_PATH)
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