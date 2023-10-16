import cv2
import os

from readimgs_renoir import read_renoir
from readimgs_sidd import read_sidd

def load_datasets(root_folder, num_images=0):
    print('Loading datasets...')
    train_folder = os.path.join(root_folder, 'train')
    test_folder = os.path.join(root_folder, 'test')
    val_folder = os.path.join(root_folder, 'val')

    renoir_train_folders = [os.path.join(train_folder, 'renoir', 'Mi3_Aligned'), 
                      os.path.join(train_folder, 'renoir', 'S90_Aligned'), 
                      os.path.join(train_folder, 'renoir', 'T3i_Aligned')]
    renoir_test_folders = [os.path.join(test_folder, 'renoir', 'Mi3_Aligned'),
                        os.path.join(test_folder, 'renoir', 'S90_Aligned'),
                        os.path.join(test_folder, 'renoir', 'T3i_Aligned')]
    renoir_val_folders = [os.path.join(val_folder, 'renoir', 'Mi3_Aligned'),
                        os.path.join(val_folder, 'renoir', 'S90_Aligned'),
                        os.path.join(val_folder, 'renoir', 'T3i_Aligned')]

    X_train, y_train = read_renoir(renoir_train_folders, num_images=num_images)
    X_test, y_test = read_renoir(renoir_test_folders, num_images=num_images)
    X_val, y_val = read_renoir(renoir_val_folders, num_images=num_images)

    sidd_train_folder = os.path.join(train_folder, 'sidd')
    sidd_test_folder = os.path.join(test_folder, 'sidd')
    sidd_val_folder = os.path.join(val_folder, 'sidd')

    X_train_sidd, y_train_sidd = read_sidd(sidd_train_folder, num_images=num_images)
    X_test_sidd, y_test_sidd = read_sidd(sidd_test_folder, num_images=num_images)
    X_val_sidd, y_val_sidd = read_sidd(sidd_val_folder, num_images=num_images)

    X_train.extend(X_train_sidd)
    y_train.extend(y_train_sidd)
    X_test.extend(X_test_sidd)
    y_test.extend(y_test_sidd)
    X_val.extend(X_val_sidd)
    y_val.extend(y_val_sidd)

    print('Loading complete.')
    return X_train, y_train, X_test, y_test, X_val, y_val
