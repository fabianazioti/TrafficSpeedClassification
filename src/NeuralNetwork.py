'''
Created on 02/08/2016

@author: fabiana
'''


import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


def sigmoid(z):
   g = 1.0/(1.0 + np.exp(-z))
   return g

def sigmoid_prime(z):
   gradientOfSigmoidFunction = sigmoid(z) * (1.0 - sigmoid(z))
   return gradientOfSigmoidFunction

class MultiLayerPerception(object):
    """
        Neural Network MultiLayerPerception
    """
    def __init__(self, sizeInput, sizeHidden, sizeOutput, lambd):
        '''
            Constructor da Classe
            MLP com uma camada Intermediaria
        '''
        self.Size_input = sizeInput
        self.Size_hidden = sizeHidden
        self.Size_output = sizeOutput
        self.Lambd = lambd
#         self. inicia_Theta_Random()
        self.inicia_Theta_Treinada()

    def inicia_Theta_Random(self):

        e = 0.12082441866603537
        self.Theta1 = (2.0 * e) * np.random.rand((self.Size_input + 1), self.Size_hidden) - e
        self.Theta2 = (2.0 * e) * np.random.rand((self.Size_hidden + 1), self.sizeOutput) - e

    def inicia_Theta_Treinada(self):
        self.Theta1 = np.loadtxt('../res/pesotheta1')
        self.Theta2 = np.loadtxt('../res/pesotheta2')

    def adiciona_Bias_Input(self, inputX):
        bias = np.ones((inputX.shape[0], 1))
        x = np.hstack((bias, inputX))
        return x

    def propagation_Values(self, inputX):

        X = self.adiciona_Bias_Input(inputX)

        a1 = X

        z2 = np.dot(a1, self.Theta1)
        a2 = sigmoid(z2)

        a2 = self.adiciona_Bias_Input(a2)
        z3 = np.dot(a2, self.Theta2)
        a3 = sigmoid(z3)

        return a1, z2, a2, z3, a3

    def getParams(self):
        return np.concatenate((self.Theta1.ravel(), self.Theta2.ravel()))

    def setParams(self, params):
        theta1_start = 0
        theta1_end = self.sizeHidden * (self.Size_input + 1)

        self.Theta1 = np.reshape(params[theta1_start:theta1_end], \
                        (self.Size_input + 1, self.Size_hidden))

        theta2_end = theta1_end + (self.Size_hidden + 1) * self.Size_output
        self.Theta2 = np.reshape(params[theta1_end:theta2_end], \
                        (self.Size_hidden + 1, self.Size_output))

    def reshape(self, params):
        theta1_start = 0
        theta1_end = self.Size_hidden * (self.Size_input + 1)

        theta1 = np.reshape(params[theta1_start:theta1_end], \
                            (self.Size_input + 1, self.Size_hidden))

        theta2_end = theta1_end + (self.Size_hidden + 1) * self.Size_output
        theta2 = np.reshape(params[theta1_end:theta2_end], \
                            (self.Size_hidden + 1, self.Size_Output))

        return theta1, theta2
