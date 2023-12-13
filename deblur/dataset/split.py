from PIL import Image
import os
import numpy as np

def ajustar_dimensoes(dimensao):
    # Garante que a dimensão seja par
    return dimensao - (dimensao % 2)

def dividir_imagem_e_salvar(imagem_path, modo, output_path):
    imagem = Image.open(imagem_path)

    largura, altura = imagem.size

    # Calcula as dimensões ajustadas das partes divididas
    metade_largura = ajustar_dimensoes(largura // 2)
    metade_altura = ajustar_dimensoes(altura // 2)

    if modo == 0:
        # Modo 0: Dividir ao meio na maior dimensão
        if largura >= altura:
            metade1 = imagem.crop((0, 0, metade_largura, altura))
            metade2 = imagem.crop((largura - metade_largura, 0, largura, altura))
        else:
            metade1 = imagem.crop((0, 0, largura, metade_altura))
            metade2 = imagem.crop((0, altura - metade_altura, largura, altura))
    elif modo == 1:
        # Modo 1: Dividir ao meio em ambas as dimensões
        metade1 = imagem.crop((0, 0, metade_largura, metade_altura))
        metade2 = imagem.crop((largura - metade_largura, 0, largura, metade_altura))
        metade3 = imagem.crop((0, altura - metade_altura, metade_largura, altura))
        metade4 = imagem.crop((largura - metade_largura, altura - metade_altura, largura, altura))
    else:
        raise ValueError("O modo deve ser 0 ou 1")

    # Salva as partes divididas
    metade1.save(os.path.join(output_path, f"split0.jpg"))
    metade2.save(os.path.join(output_path, f"split1.jpg"))

    if modo == 1:
        metade3.save(os.path.join(output_path, f"split2.jpg"))
        metade4.save(os.path.join(output_path, f"split3.jpg"))

    # Exibe as dimensões das partes
    print(f"Dimensões da metade1: {metade1.size}")
    print(f"Dimensões da metade2: {metade2.size}")
    if modo == 1:
        print(f"Dimensões da metade3: {metade3.size}")
        print(f"Dimensões da metade4: {metade4.size}")

    return metade1, metade2, metade3, metade4 if modo == 1 else None


def dividir_imagem(imagem, modo=1):
   # imagem = Image.fromarray(imagem.astype('uint8'), 'RGB')

    largura, altura = imagem.size

    # Calcula as dimensões ajustadas das partes divididas
    metade_largura = ajustar_dimensoes(largura // 2)
    metade_altura = ajustar_dimensoes(altura // 2)

    if modo == 0:
        # Modo 0: Dividir ao meio na maior dimensão
        if largura >= altura:
            metade1 = imagem.crop((0, 0, metade_largura, altura))
            metade2 = imagem.crop((largura - metade_largura, 0, largura, altura))
        else:
            metade1 = imagem.crop((0, 0, largura, metade_altura))
            metade2 = imagem.crop((0, altura - metade_altura, largura, altura))
    elif modo == 1:
        # Modo 1: Dividir ao meio em ambas as dimensões
        metade1 = imagem.crop((0, 0, metade_largura, metade_altura))
        metade2 = imagem.crop((largura - metade_largura, 0, largura, metade_altura))
        metade3 = imagem.crop((0, altura - metade_altura, metade_largura, altura))
        metade4 = imagem.crop((largura - metade_largura, altura - metade_altura, largura, altura))
    else:
        raise ValueError("O modo deve ser 0 ou 1")

    # Salva as partes divididas
#    metade1.save(os.path.join(output_path, f"split0.jpg"))
#    metade2.save(os.path.join(output_path, f"split1.jpg"))

#    if modo == 1:
#        metade3.save(os.path.join(output_path, f"split2.jpg"))
#        metade4.save(os.path.join(output_path, f"split3.jpg"))

    # Exibe as dimensões das partes
#    print(f"Dimensões da metade1: {metade1.size}")
#    print(f"Dimensões da metade2: {metade2.size}")
#    if modo == 1:
#        print(f"Dimensões da metade3: {metade3.size}")
#        print(f"Dimensões da metade4: {metade4.size}")

#    return np.array(metade1), np.array(metade2), np.array(metade3), np.array(metade4) if modo == 1 else None

    return metade1, metade2, metade3, metade4 if modo == 1 else None

# Exemplo de uso
#modo = 1  # ou 0
#imagem_path = 'C:/Users/Danilo/Downloads/Uformer-main/dataset/deblurring/test/input/A.png'
#output_path = 'C:/Users/Danilo/Downloads/Uformer-main/dataset/deblurring/test/input'

#try:
#    if not os.path.exists(output_path):
#        os.makedirs(output_path)
#
#    resultado_divisao = dividir_imagem_e_salvar(imagem_path, modo, output_path)
#
#    if resultado_divisao is not None:
#        metade1, metade2, metade3, metade4 = resultado_divisao
#except ValueError as e:
#    print(e)
#except TypeError:
#    # Trata o caso em que a função retorna None
#    metade1, metade2, metade3, metade4 = None, None, None, None
