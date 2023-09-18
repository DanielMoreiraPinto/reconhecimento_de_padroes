import os

import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from readimgs import read_renoir
from networks import simple_autoencoder, cbd_net, rid_net

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

class Dataloader(tf.keras.utils.Sequence):    
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
        # normalize data
        batch_x = batch_x / 255.0
        batch_y = batch_y / 255.0
        
        return tuple((batch_x,batch_y))
    
    def __len__(self):
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

def preprocess_data(X, y, shuffle):
    # Shuffle the data if specified
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

    # Normalize the data
    X = X / 255.0
    y = y / 255.0

    return X, y

def train_model(X_train, y_train, X_val, y_val, model_type='simple_autoencoder', model_path='data\\denoiser.h5', epochs=10, batch_size=32):
    # Create dataloaders for the training, validation and testing sets
    train_dataloader = Dataloader(X_train, y_train, batch_size, shuffle=True)
    val_dataloader = Dataloader(X_val, y_val, batch_size, shuffle=True)

    # Create the model
    model = select_model(model_type)

    # Create validation loss checkpoint
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    checkpoint = ModelCheckpoint(model_path,
                                monitor='val_loss',
                                mode='min',
                                save_best_only=True,
                                verbose=1,
                                save_weights_only=True)

    print("Training model...")
    model.fit(train_dataloader, epochs=epochs, validation_data=val_dataloader, callbacks=[checkpoint])
    print("Training complete.")

    return model

def select_model(model_type):
    if model_type == 'simple_autoencoder':
        model = simple_autoencoder()
    elif model_type == 'cbd_net':
        model = cbd_net()
    elif model_type == 'rid_net':
        model = rid_net()
    else:
        raise ValueError("Invalid model type. Valid model types are 'simple_autoencoder', 'cbd_net' and 'rid_net'.")
    return model

def load_model(model_path, model_type='simple_autoencoder'):
    print("Loading model...")
    model = select_model(model_type)
    model.load_weights(model_path)
    print("Model loaded.")
    return model

def test_model(model, X_test, y_test):
    # Predict with the model all images in the testing set
    print("Testing model...")
    y_pred = model.predict(X_test/255.0)
    y_pred = y_pred * 255.0
    y_pred = y_pred.astype(np.uint8)

    # Calculate the PSNR and SSIM for each image
    psnr_list = []
    ssim_list = []
    for i in range(len(y_test)):
        psnr_list.append(psnr(y_test[i], y_pred[i]))
        ssim_list.append(ssim(y_test[i], y_pred[i], multichannel=True))

    # Calculate the average PSNR and SSIM
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print("Testing complete.")
    print(f"Average PSNR: {avg_psnr}")
    print(f"Average SSIM: {avg_ssim}")

    metrics_by_image = [psnr_list, ssim_list]
    avg_metrics = [avg_psnr, avg_ssim]

    return y_pred, metrics_by_image, avg_metrics


RENOIR_DATASET_PATHS = ['D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\Mi3_Aligned',
                        'D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\S90_Aligned',
                        'D:\\daniel_moreira\\reconhecimento_de_padroes\\bases\\T3i_Aligned']
MODEL_TYPE = 'simple_autoencoder'
MODEL_PATH = f'D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\denoiser\\data\\{MODEL_TYPE}.h5'
EPOCHS = 40
BATCH_SIZE = 32

def experiments(ckpt_path = None):
    # Read the images from the dataset
    X, y = read_renoir(RENOIR_DATASET_PATHS)

    # Divide the dataset into training, validation and testing sets
    # 80% training, 10% validation, 10% testing
    X_train, y_train = X[:int(len(X)*0.8)], y[:int(len(y)*0.8)]
    X_val, y_val = X[int(len(X)*0.8):int(len(X)*0.9)], y[int(len(y)*0.8):int(len(y)*0.9)]
    X_test, y_test = X[int(len(X)*0.9):], y[int(len(y)*0.9):]

    # Create patches from the images
    X_train_patches, y_train_patches = patchify(X_train, y_train)
    X_val_patches, y_val_patches = patchify(X_val, y_val)
    X_test_patches, y_test_patches = patchify(X_test, y_test)

    aux1 = X_train_patches.shape
    aux2 = y_train_patches.shape
    aux3 = X_val_patches.shape
    aux4 = y_val_patches.shape
    aux5 = X_test_patches.shape
    aux6 = y_test_patches.shape
    
    # Train the model
    if ckpt_path is not None:
        model = load_model(ckpt_path, model_type=MODEL_TYPE)
    else:
        model = train_model(X_train_patches, y_train_patches, X_val_patches, y_val_patches, model_type=MODEL_TYPE, model_path=MODEL_PATH, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Test the model
    test_model(model, X_test_patches, y_test_patches)


if __name__ == "__main__":
    experiments(ckpt_path=None)
