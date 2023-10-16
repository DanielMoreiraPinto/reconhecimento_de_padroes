# from networks import simple_autoencoder, cbd_net, rid_net, dn_cnn

# model = simple_autoencoder()
# model = cbd_net()
# model = rid_net()
# model = dn_cnn()



# import patchify
# import cv2 as cv
# import os

# # img_path = 'C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\denoiser\\data\\RENOIR\\Mi3_Aligned\\Batch_001\\IMG_20160202_015216Reference.jpg'
# # img_path = 'C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\denoiser\\data\\SIDD_Medium_Srgb\\Data\\0088_004_IP_00100_00050_5500_N\\0088_NOISY_SRGB_010.jpg'
# img_path = 'C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\denoiser\\data\\RENOIR\\Mi3_Aligned\\Batch_039\\IMG_20160224_053659Reference.jpg'
# save_patches_path = 'C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\denoiser\\data\\test\\patches'
# save_reconstructed_path = 'C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\denoiser\\data\\test\\reconstructed.jpg'

# img = cv.imread(img_path)
# orig_shape = img.shape
# patches = patchify.split_image(img, patch_size=256)
# for i, patch in enumerate(patches):
#     cv.imwrite(os.path.join(save_patches_path, str(i)+'.jpg'), patch)
    
# reconstructed = patchify.reconstruct_image(patches, orig_shape)
# cv.imwrite(save_reconstructed_path, reconstructed)



# from readimgs_sidd import read_sidd
# import os
# import cv2 as cv

# root_folder = 'C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\denoiser\\data\\SIDD_Medium_Srgb\\Data'
# targets, labels = read_sidd(root_folder, num_images=40)
# x = 1
# # print the images
# output_folder = 'C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\denoiser\\data\\test\\sidd'
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
# for i in range(len(targets)):
#     cv.imwrite(os.path.join(output_folder, str(i)+'_target_'+'.jpg'), targets[i])
#     cv.imwrite(os.path.join(output_folder, str(i)+'_label_'+'.jpg'), labels[i])

#  I have numbers from 1 to 40. I want to put 60% of them in one group, 20% in another and
# 20% in another.
import numpy as np
import random

numbers = np.arange(1, 41)
random.shuffle(numbers)
print(numbers)
print(len(numbers))
print(sorted(numbers[:24]))
print(sorted(numbers[24:32]))
print(sorted(numbers[32:]))