# import sys
# sys.path.append('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes')

import os
import pickle

import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from read_datasets import load_datasets
from networks import simple_autoencoder, cbd_net, rid_net, dn_cnn
from patchify import split_image, reconstruct_image

np.random.seed(42)


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

# def preprocess_data(X, y, shuffle):
#     # Shuffle the data if specified
#     if shuffle:
#         indices = np.arange(len(X))
#         np.random.shuffle(indices)
#         X = X[indices]
#         y = y[indices]

#     # Normalize the data
#     X = (X / 255.0).astype(np.float32)
#     y = (y / 255.0).astype(np.float32)

#     return X, y

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=4, shuffle=True, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        self.X = X
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size

    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.X))
            np.random.shuffle(indices)
            self.X = self.X[indices]
            self.y = self.y[indices]

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        X_batch = self.X[index*self.batch_size:(index+1)*self.batch_size]
        y_batch = self.y[index*self.batch_size:(index+1)*self.batch_size]
        X_batch = (X_batch / 255.0)#.astype(np.float32)
        y_batch = (y_batch / 255.0)#.astype(np.float32)
        return X_batch, y_batch

def train_model(X_train, y_train, X_val, y_val, model_type='simple_autoencoder', 
                model_path='data\\denoiser.h5', epochs=10, batch_size=32):
    # Create the model
    model = select_model(model_type)

    # Create validation loss checkpoint
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # Create validation loss early stop
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

    train_gen = DataGenerator(X_train, y_train, batch_size=batch_size, 
                              shuffle=True, random_state=42)
    val_gen = DataGenerator(X_val, y_val, batch_size=batch_size, 
                            shuffle=True, random_state=42)

    print("Training model...")
    model.fit(train_gen, batch_size=batch_size, epochs=epochs, 
              validation_data=val_gen, 
              callbacks=[checkpoint, early_stop])
    print("Training complete.")
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
    # X_test, y_test = preprocess_data(X_test, y_test, shuffle=False)

    # Calculate the PSNR and SSIM for each image
    psnr_noisy_list = []
    ssim_noisy_list = []
    psnr_list = []
    ssim_list = []
    for i in range(len(y_test)):
        X, y, y_pred = pred_image(model, X_test[i], y_test[i])
        psnr_list.append(psnr(y, y_pred))
        ssim_list.append(ssim(y, y_pred, channel_axis=-1))
        psnr_noisy_list.append(psnr(y, X))
        ssim_noisy_list.append(ssim(y, X, channel_axis=-1))

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

    return metrics_by_image, avg_metrics

def pred_image(model, X_test, y_test):
    X_test = X_test / 255.0
    X_test = np.expand_dims(X_test, axis=0)
    y_pred = model.predict(X_test)
    y_pred = y_pred.squeeze()
    X_test = X_test.squeeze()
    X_test = X_test * 255.0
    X_test = X_test.astype(np.uint8)
    y_pred = y_pred * 255.0
    y_pred = y_pred.astype(np.uint8)
    y_test = y_test * 255.0
    y_test = y_test.astype(np.uint8)
    return X_test,y_test,y_pred

def patchify(X, y, patch_ratio):
    X_patches, y_patches = [], []
    for i in range(len(X)):
        # each i is a path to an image, i need to read the image
        image = cv.imread(X[i])
        target = cv.imread(y[i])
        X_patches_image, y_patches_image = split_image(image, target, 256, patch_ratio)
        image, target = None, None
        X_patches.extend(X_patches_image)
        y_patches.extend(y_patches_image)
    return np.array(X_patches), np.array(y_patches)

def training(ckpt_path = None, num_images=0, patch_ratio=0):
    # Read the images from the dataset
    X_train, y_train, X_test, y_test, X_val, y_val = load_datasets(DATASET_PATH, num_images)
    
    # Save the testing images with pickle
    if not os.path.exists(TEST_SAVE_PATH):
        os.makedirs(TEST_SAVE_PATH)
        with open(os.path.join(TEST_SAVE_PATH, 'x_test.pkl'), 'wb') as f:
            pickle.dump(X_test, f)
        with open(os.path.join(TEST_SAVE_PATH, 'y_test.pkl'), 'wb') as f:
            pickle.dump(y_test, f)
        print("Testing set saved.")
    else:
        print("Testing set already exists.")

    # Create patches from the images
    if ckpt_path is None:
        print("Creating train patches...")
        X_train, y_train = patchify(X_train, y_train, patch_ratio)
        print("Creating validation patches...")
        X_val, y_val = patchify(X_val, y_val, patch_ratio)
    print("Creating test patches...")
    X_test, y_test = patchify(X_test, y_test, patch_ratio)
    print("Patches created.")
    
    # Train the model
    if ckpt_path is not None:
        # Load the model
        model = load_model_weights(ckpt_path, model_type=MODEL_TYPE)
        # Test the model
        test_model(model, X_test, y_test)
    else:
        model = train_model(X_train, y_train, X_val, y_val, model_type=MODEL_TYPE, 
                            model_path=MODEL_PATH, epochs=EPOCHS, 
                            batch_size=BATCH_SIZE)
        # # Test the last model
        # test_model(model, X_test, y_test)
        # Test the best model
        model = load_model_weights(MODEL_PATH, model_type=MODEL_TYPE)
        test_model(model, X_test, y_test)

def denoise(image):
    # Load the model
    model = load_model_weights(MODEL_PATH, model_type=MODEL_TYPE)
    # Preprocess the image
    img_shape = image.shape
    denoised = split_image(image, patch_size=256)
    denoised = denoised / 255.0
    # Denoise the image
    denoised = model.predict(denoised)
    denoised = denoised * 255.0
    denoised = denoised.astype(np.uint8)
    denoised = reconstruct_image(denoised, img_shape)
    return denoised

def test_denoising(test_path, save_path):
    print("Loading testing set...")
    X_test, y_test = [], []
    with open(os.path.join(test_path, 'x_test.pkl'), 'rb') as f:
        x_test_paths = pickle.load(f)
        for path in x_test_paths:
            X_test.append(cv.imread(path))
    with open(os.path.join(test_path, 'y_test.pkl'), 'rb') as f:
        y_test_paths = pickle.load(f)
        for path in y_test_paths:
            y_test.append(cv.imread(path))
    print("Testing set loaded.")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    y_pred = []
    for i in range(len(X_test)):
        pred = denoise(X_test[i])
        cv.imwrite(os.path.join(save_path, f'{i}_noise.png'), X_test[i])
        cv.imwrite(os.path.join(save_path, f'{i}_true.png'), y_test[i])
        cv.imwrite(os.path.join(save_path, f'{i}_pred.png'), pred)
        if i == 19:
            break
    print("Testing complete.")

def test_image():
    img = cv.imread('D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\RENOIR\\T3i_Aligned\\Batch_032\\IMG_7729Reference.jpg')
    img = denoise(img)
    cv.imwrite('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\denoiser\\data\\tests\\_teste_unico.png', img)
    

# RENOIR_DATASET_PATHS = ['D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\RENOIR\\Mi3_Aligned',
#                         'D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\RENOIR\\S90_Aligned',
#                         'D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\RENOIR\\T3i_Aligned']
# SIDD_DATASET_PATH = 'D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\SIDD_Medium_Srgb\\Data'
DATASET_PATH = 'D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\dataset'
TEST_SAVE_PATH = 'D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\test'
TEST_SAMPLES_PATH = 'D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\denoiser\data\\test_sample'

MODEL_TYPE = 'simple_autoencoder'
# MODEL_TYPE = 'cbd_net'
# MODEL_TYPE = 'rid_net'
# MODEL_TYPE = 'dn_cnn'
MODEL_PATH = f'D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\denoiser\\data\\models\\{MODEL_TYPE}.h5'
# MODEL_PATH = f'D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\denoiser\\data\\models\\simple_autoencoder\\simple_autoencoder.h5'

EPOCHS = 100
BATCH_SIZE = 16

if __name__ == "__main__":
    print("Iniciando...")
    # training(num_images=0, patch_ratio=0.3)
    training(ckpt_path=MODEL_PATH, patch_ratio=0.3)
    # test_denoising(TEST_SAVE_PATH, TEST_SAMPLES_PATH)
    # test_image()
    pass
