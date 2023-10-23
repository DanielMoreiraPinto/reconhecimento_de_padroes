import setup_path
import cv2 as cv

from denoiser.denoiser import training, test_denoising, denoise, smooth_image_lines

if __name__ == "__main__":
    # training()
    
    # training(ckpt_path=MODEL_PATH)
    
    # test_denoising(TEST_SAVE_PATH, TEST_SAMPLES_PATH)
    
    # image = cv.imread("C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\IMG_7830Noisy.jpg")
    # image = denoise(image)
    # cv.imwrite("C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\IMG_7830Denoised.jpg", image)
    
    image = cv.imread("C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\IMG_7830Denoised.jpg")
    image = smooth_image_lines(image)
    cv.imwrite("C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\IMG_7830Smooth.jpg", image)
    pass