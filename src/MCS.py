'''
Created on 01/08/2016

@author: fabiana
'''

import numpy as np
import cv2
import random
from NeuralNetwork import MultiLayerPerception

TAMANHO_ENTRADA = 400
TAMANHO_ESCONDIDA = 25
TAMANHO_SAIDA = 10
LAMBDA = 1.0

class ClassficacaoContorno(object):
    '''
        Realiza a classificacao dos contornos
    '''

    def __init__(self, numero):
        '''
        Construtor da classe
        '''
        self.Numero = numero
        self.Vetor_img = None

    def binarizacao(self):
        self.Numero = cv2.cvtColor(self.Numero, cv2.COLOR_BGR2GRAY)
        _,self.Numero = cv2.threshold(self.Numero, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    def normalizar(self):
        self.Numero = cv2.resize(self.Numero,(20,20))
        self.Vetor_img = np.reshape(self.Numero,(1,400))
        #self.vetorImg = self.vetorImg.astype('uint8')

    def classificarN(self):
        self.binarizacao()
        self.normalizar()

        MLP = MultiLayerPerception(TAMANHO_ENTRADA, TAMANHO_ESCONDIDA, TAMANHO_SAIDA, LAMBDA)
        # print'\n\nPropagando Valores ..  \n'

        _, _, _,_, z3 =  MLP.propagation_Values(self.Vetor_img)
        z3 = np.round(z3, decimals = 4)
        # print '\nResultado: \n', z3

        #print'\nResultado: \n', np.round(z3, decimals = 4)

        maxX = z3.argmax()
        # print '\nClasse ', maxX
        return maxX
    def prepararBase(self, cont):
        self.threshold()
        self.reshape()

        nomeSalvarImagem = '/home/detect_%d'%cont
        np.savetxt(nomeSalvarImagem, self.vetorImg.astype('float64'), fmt='%1.3e')
