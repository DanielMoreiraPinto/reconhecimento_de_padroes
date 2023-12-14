import setup_path
from superesolution.supres_class import aumentar_resolucao
import cv2

# img = aumentar_resolucao('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\superesolution\\inputs\\shanghai.jpg')
# cv2.imwrite('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\superesolution\\results\\shanghai.jpg', img)

from deblur.test.testerClass import chamar_deblur, chamar_denoiser

# img = chamar_deblur('C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\ex_input\\GOPR0384_11_00-000001.png')
# cv2.imwrite('C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\ex_output\\GOPR0384_11_00-000001.png', img)
# img = chamar_deblur('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\ex_input\\GOPR0384_11_00-000001.png')
# cv2.imwrite('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\ex_output\\GOPR0384_11_00-000001.png', img)
# img = chamar_denoiser('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\ex_input\\IMG_20160224_053715Noisy.jpg')
# cv2.imwrite('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\ex_output\\IMG_20160224_053715Noisy.jpg', img)
# img = chamar_deblur('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\ex_input\\IMG_20160224_053715Noisy_denoised.jpg')
# cv2.imwrite('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\ex_output\\IMG_20160224_053715Noisy_denoised.jpg', img)

from blurdetect.detect_blur import BlurDetector

# img = cv2.imread('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\ex_input\\3_HUAWEI-NOVA-LITE_M.jpg')
img = cv2.imread('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\ex_input\\3_HUAWEI-NOVA-LITE_F.jpg')
detector = BlurDetector()
print(detector.detect_blur(img))
img = chamar_deblur('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\ex_input\\8_SAMSUNG-GALAXY-A6_F.jpg')
cv2.imwrite('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\ex_output\\8_SAMSUNG-GALAXY-A6_F.jpg', img)