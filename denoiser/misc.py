# from networks import simple_autoencoder, cbd_net, rid_net, dn_cnn

# model = simple_autoencoder()
# model = cbd_net()
# model = rid_net()
# model = dn_cnn()


import patchify
import cv2 as cv
import os

# img_path = 'C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\denoiser\\data\\RENOIR\\Mi3_Aligned\\Batch_001\\IMG_20160202_015216Reference.jpg'
# img_path = 'C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\denoiser\\data\\SIDD_Medium_Srgb\\Data\\0088_004_IP_00100_00050_5500_N\\0088_NOISY_SRGB_010.jpg'
img_path = 'C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\denoiser\\data\\RENOIR\\Mi3_Aligned\\Batch_039\\IMG_20160224_053659Reference.jpg'
save_patches_path = 'C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\denoiser\\data\\test\\patches'
save_reconstructed_path = 'C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\denoiser\\data\\test\\reconstructed.jpg'

img = cv.imread(img_path)
orig_shape = img.shape
patches = patchify.split_image(img, patch_size=256)
for i, patch in enumerate(patches):
    cv.imwrite(os.path.join(save_patches_path, str(i)+'.jpg'), patch)
    
reconstructed = patchify.reconstruct_image(patches, orig_shape)
cv.imwrite(save_reconstructed_path, reconstructed)