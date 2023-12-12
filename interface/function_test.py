import setup_path
from superesolution.supres_class import aumentar_resolucao
import cv2

img = aumentar_resolucao('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\superesolution\\inputs\\shanghai.jpg')
cv2.imwrite('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\superesolution\\results\\shanghai.jpg', img)