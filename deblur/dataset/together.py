from PIL import Image
import os
import numpy as np

def unir_imagens_por_caminho(imagem1_path, imagem2_path, imagem3_path=None, imagem4_path=None, output_path=None):
    imagens = [Image.open(imagem1_path), Image.open(imagem2_path)]

    if imagem3_path is not None:
        imagens.append(Image.open(imagem3_path))

    if imagem4_path is not None:
        imagens.append(Image.open(imagem4_path))

    # Verifica as dimensões
    if not all(imagens[0].size == img.size for img in imagens):
        raise ValueError("As dimensões das imagens não são compatíveis")

    # Obtém as dimensões da imagem unida
    largura, altura = imagens[0].size
    imagem_unida = Image.new("RGB", (largura * 2, altura * 2))

    # Une as imagens
    for i, img in enumerate(imagens):
        if i == 0:
            imagem_unida.paste(img, (0, 0))
        elif i == 1:
            imagem_unida.paste(img, (largura, 0))
        elif i == 2:
            imagem_unida.paste(img, (0, altura))
        elif i == 3:
            imagem_unida.paste(img, (largura, altura))

    # Salva a imagem unida se um caminho de saída foi fornecido
    if output_path is not None:
        imagem_unida.save(output_path)

    return imagem_unida


def unir_imagens(images):
    import cv2 as cv
    # for i, img in enumerate(images):
    #     cv.imwrite(str(i)+'.png', img)    
    imagens = [Image.fromarray(pixels.astype('uint8'), 'RGB') for pixels in images]
    
    # Verifica as dimensões
    if not all(imagens[0].size == img.size for img in imagens):
        raise ValueError("As dimensões das imagens não são compatíveis")

    # Obtém as dimensões da imagem unida
    largura, altura = imagens[0].size
    imagem_unida = Image.new("RGB", (largura * 2, altura * 2))

    # Une as imagens
    for i, img in enumerate(imagens):
        if i == 0:
            imagem_unida.paste(img, (0, 0))
        elif i == 1:
            imagem_unida.paste(img, (largura, 0))
        elif i == 2:
            imagem_unida.paste(img, (0, altura))
        elif i == 3:
            imagem_unida.paste(img, (largura, altura))

#    # Salva a imagem unida se um caminho de saída foi fornecido
#    if output_path is not None:
#        imagem_unida.save(output_path)

    return np.array(imagem_unida)

## Exemplo de uso
#imagem1_path = 'C:/Users/Danilo/Downloads/Uformer-main/dataset/deblurring/test/input/split0.jpg'
#imagem2_path = 'C:/Users/Danilo/Downloads/Uformer-main/dataset/deblurring/test/input/split1.jpg'
#imagem3_path = 'C:/Users/Danilo/Downloads/Uformer-main/dataset/deblurring/test/input/split2.jpg'  # opcional
#imagem4_path = 'C:/Users/Danilo/Downloads/Uformer-main/dataset/deblurring/test/input/split3.jpg'  # opcional
#output_path = 'C:/Users/Danilo/Downloads/Uformer-main/dataset/deblurring/test/input/union.jpg'
#
## Chama a função para unir as imagens
#try:
#    imagem_unida = unir_imagens(imagem1_path, imagem2_path, imagem3_path, imagem4_path, output_path)
#    imagem_unida.show()  # Se quiser exibir a imagem
#
#except ValueError as e:
#    print(e)
#