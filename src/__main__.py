'''
Created on 09/08/2016

@author: fabiana
'''

# from picamera.array import PiRGBArray
# from picamera import PiCamera

import cv2
import numpy as np
from matplotlib import pyplot as plt
from MDSV import DetectarPlacas

def classificarSinImagem():
    Numero_imagens = 10
    Posicao_Atual = 1

    while Posicao_Atual <= Numero_imagens:
        caminhoImagem = '/%d.png'%Posicao_Atual
        print '\n# ----------------- IMAGEM %d ------------------#\n'%Posicao_Atual
        Detec_sinV = DetectarPlacas(cv2.imread(caminhoImagem))
        Detec_sinV.segmentacao_Cor()
        Detec_sinV.detectar_Circulo()

        if Detec_sinV.Posicao_circulos is not None:
            Detec_sinV.regioes_Interesse()

            Detec_sinV.desenha_Resultado_IMG()

        Posicao_Atual += 1



if __name__ == '__main__':
    classificarSinImagem()
