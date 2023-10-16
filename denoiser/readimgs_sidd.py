import os
import glob
import random
import cv2 as cv


def find_gt_and_noisy_images(folder_path):
    # List all files in the folder
    files = glob.glob(os.path.join(folder_path, '*.jpg'))
    
    gt, noisy = None, None
    for file in files:
        if 'GT' in os.path.basename(file).upper() and gt is None:
            gt = file
        elif 'NOISY' in os.path.basename(file).upper() and noisy is None:
            noisy = file
    
    # Return one GT and one NOISY image if available
    return gt, noisy


# Main function to process folders
def read_sidd(root_folder, num_images=0):
    print('Reading SIDD...')
    targets, labels = [], []
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        
        if os.path.isdir(folder_path):
            gt_image, noisy_image = find_gt_and_noisy_images(folder_path)
            
            if gt_image and noisy_image:
                targets.append(cv.imread(noisy_image))
                labels.append(cv.imread(gt_image))
        if num_images > 0 and len(targets) % num_images == 0:
            break
    print('Reading complete.')
    return targets, labels