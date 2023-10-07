import cv2
import os

# Function to find reference and noisy images in a folder
def find_images_in_folder(folder):
    reference_image = None
    noisy_image = None

    # Iterate through files in the folder
    for filename in os.listdir(folder):
        if filename.lower().endswith(".jpg"):
            file_path = os.path.join(folder, filename)

            # Check if the image contains "Reference" or "Noisy"
            if "reference" in filename.lower():
                reference_image = cv2.imread(file_path)
            elif "noisy" in filename.lower():
                noisy_image = cv2.imread(file_path)

            # Break if both images are found
            if reference_image is not None and noisy_image is not None:
                break

    return reference_image, noisy_image

# Main function to process folders
def read_renoir(root_folders, num_images=0):
    print('Reading RENOIR...')
    targets, labels = [], []
    for root_folder in root_folders:
        print('Reading folder: ', root_folder)
        for foldername in os.listdir(root_folder):
            folder_path = os.path.join(root_folder, foldername)
            if os.path.isdir(folder_path):
                reference_image, noisy_image = find_images_in_folder(folder_path)
                
                targets.append(noisy_image)
                labels.append(reference_image)
                if num_images > 0 and len(targets) % num_images == 0:
                    break
    print('Reading complete.')
    return targets, labels