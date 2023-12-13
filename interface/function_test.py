import setup_path
from superesolution.supres_class import aumentar_resolucao
import cv2

img, shape = aumentar_resolucao('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\ex_input\\shanghai.jpg')
cv2.imwrite('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\ex_output\\shanghai.jpg', img)
print(shape)

from deblur.test.testerClass import chamar_deblur

# img = chamar_deblur('C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\ex_input\\GOPR0384_11_00-000001.png')
# cv2.imwrite('C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\ex_output\\GOPR0384_11_00-000001.png', img)
# img = chamar_deblur('C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\ex_input\\GOPR0384_11_00-000013.png')
# cv2.imwrite('C:\\Users\\danie\\OneDrive\\Documentos\\Mestrado\\reconhecimento_de_padroes\\ex_output\\GOPR0384_11_00-000013.png', img)