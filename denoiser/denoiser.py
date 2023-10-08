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

from readimgs_renoir import read_renoir
from readimgs_sidd import read_sidd
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

def preprocess_data(X, y, shuffle):
    # Shuffle the data if specified
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

    # Normalize the data
    X = (X / 255.0).astype(np.float32)
    y = (y / 255.0).astype(np.float32)

    return X, y

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32):
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        X_batch = self.X[index*self.batch_size:(index+1)*self.batch_size]
        y_batch = self.y[index*self.batch_size:(index+1)*self.batch_size]
        return X_batch, y_batch

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
    # Create validation loss early stop
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

    train_gen = DataGenerator(train_data, train_label, batch_size=batch_size)
    val_gen = DataGenerator(val_data, val_label, batch_size=batch_size)

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
    X_test, y_test = preprocess_data(X_test, y_test, shuffle=False)
    X_test, y_test, y_pred = pred_images(model, X_test, y_test)

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

    return metrics_by_image, avg_metrics, y_pred, y_test

def pred_images(model, X_test, y_test):
    y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
    X_test = X_test * 255.0
    X_test = X_test.astype(np.uint8)
    y_pred = y_pred * 255.0
    y_pred = y_pred.astype(np.uint8)
    y_test = y_test * 255.0
    y_test = y_test.astype(np.uint8)
    return X_test,y_test,y_pred

def patchify(X, y):
    X_patches, y_patches = [], []
    for i in range(len(X)):
        X_patches.extend(split_image(X[i], 256))
        y_patches.extend(split_image(y[i], 256))
    return np.array(X_patches), np.array(y_patches)

def training(ckpt_path = None):
    # Read the images from the dataset
    X, y = read_renoir(RENOIR_DATASET_PATHS, num_images=10)

    # Divide the dataset into training, validation and testing sets
    # 80% training, 10% validation, 10% testing
    X_train, y_train = X[:int(len(X)*0.8)], y[:int(len(y)*0.8)]
    X_val, y_val = X[int(len(X)*0.8):int(len(X)*0.9)], y[int(len(y)*0.8):int(len(y)*0.9)]
    X_test, y_test = X[int(len(X)*0.9):], y[int(len(y)*0.9):]
    X = None
    y = None
    
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
    X_train, y_train = patchify(X_train, y_train)
    X_val, y_val = patchify(X_val, y_val)
    X_test, y_test = patchify(X_test, y_test)
    
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
    denoised = split_image(image, 256)
    denoised = denoised / 255.0
    # Denoise the image
    denoised = model.predict(denoised)
    denoised = denoised * 255.0
    denoised = denoised.astype(np.uint8)
    denoised = reconstruct_image(denoised, img_shape)
    return denoised

def test_denoising(test_path, save_path):
    print("Loading testing set...")
    with open(os.path.join(test_path, 'x_test.pkl'), 'rb') as f:
        X_test = pickle.load(f)
    with open(os.path.join(test_path, 'y_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
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


RENOIR_DATASET_PATHS = ['D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\RENOIR\\Mi3_Aligned',
                        'D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\RENOIR\\S90_Aligned',
                        'D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\RENOIR\\T3i_Aligned']
SIDD_DATASET_PATH = 'D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\SIDD_Medium_Srgb\\Data'
TEST_SAVE_PATH = 'D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\test'
TEST_SAMPLES_PATH = 'D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\denoiser\data\\test_sample'

MODEL_TYPE = 'simple_autoencoder'
# MODEL_TYPE = 'cbd_net'
# MODEL_TYPE = 'rid_net'
# MODEL_TYPE = 'dn_cnn'
MODEL_PATH = f'D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\denoiser\\data\\models\\{MODEL_TYPE}.h5'
# MODEL_PATH = f'D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\denoiser\\data\\models\\v2\\{MODEL_TYPE}.h5'

EPOCHS = 100
BATCH_SIZE = 32

if __name__ == "__main__":
    # training()
    # training(ckpt_path=MODEL_PATH)
    test_denoising(TEST_SAVE_PATH, TEST_SAMPLES_PATH)
    pass
