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
        self. inicia_Theta_Random()
        # self.inicia_Theta_Treinada()

    def inicia_Theta_Random(self):

        e = 0.12082441866603537
        self.Theta1 = (2.0 * e) * np.random.rand((self.Size_input + 1), self.Size_hidden) - e
        self.Theta2 = (2.0 * e) * np.random.rand((self.Size_hidden + 1), self.Size_output) - e

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
                            (self.Size_hidden + 1, self.Size_output))

        return theta1, theta2

class TrainerMLP(object):
    """docstring for TrainerN"""
    def __init__(self, RN, X, y, testX, testY):
        self.RN = RN
        self.y = y
        self.X = X
        self.m = X.shape[0]
        self.testX = testX
        self.testY = testY
        self.testM = testX.shape[0]

    def costFunciNN(self, X, y, m):

        _, _, _, _, h0 = self.RN.propagation_Values(X)
        J = -(1.0 / m) * sum(sum(y * np.log(h0) + (1.0 - y) * np.log(1.0 - h0)))

        reg = (self.RN.Lambd / (2.0 * m)) * (sum(sum(self.RN.Theta1[1:] ** 2)) + sum(sum(self.RN.Theta2[1:] ** 2)))
        J = J + reg

        return J


    def partialDeriv(self):

        self.Theta1_grad = np.zeros(self.RN.Theta1.shape)
        self.Theta2_grad = np.zeros(self.RN.Theta2.shape)

        G1 = np.zeros(self.RN.Theta1.shape)
        G2 = np.zeros(self.RN.Theta2.shape)


        for i in xrange(self.m):
            inputX = np.array(([self.X[i]]), dtype=float)
            a1, z2, a2, z3, ra3 = self.RN.propagation_Values(inputX)

            err3 = np.array(ra3 - self.y[i])
            errL2 = np.dot(err3, self.RN.Theta2.T)
            err2 = errL2[:, 1:] * sigmoid_prime(z2)
            G2 = G2 + np.dot(a2.T, err3)
            G1 = G1 + np.dot(a1.T, err2)


        DT1B = G1[:1] / self.m
        DT1 = (G1[1:] / self.m) + (self.RN.Lambd * self.RN.Theta1[1:] / self.m)
        self.Theta1_grad = np.concatenate((DT1B, DT1))

        DT2B = G2[:1] / self.m
        DT2 = (G2[1:] / self.m) + (self.RN.Lambd * self.RN.Theta2[1:] / self.m)

        self.Theta2_grad = np.concatenate((DT2B, DT2))

        grad = np.concatenate((self.Theta1_grad.ravel(), self.Theta2_grad.ravel()))

        return grad

    def gradientDescent(self, epochs, learning_rate):
        cost = []
        costTreiner = []
        for interacoes in range(epochs):
            grad = self.partialDeriv()
            Gradiente1, Gradiente2 = self.RN.reshape(grad)
            self.RN.Theta1 += -learning_rate * Gradiente1
            self.RN.Theta2 += -learning_rate * Gradiente2

            J = self.costFunciNN(self.X, self.y, self.m)
            J_treiner = self.costFunciNN(self.testX, self.testY, self.testM)

            cost.append(J)
            costTreiner.append(J_treiner)
            print 'Interacao %d' % interacoes, '||' , 'Custo .. %1.3e ' % J

        plt.plot(cost, label="Custo")
        plt.plot(costTreiner, label="Custo Treinamento")
        plt.legend()
        plt.grid(1)
        plt.xlabel('Numero Iteracoes')
        plt.ylabel('Custo')
        plt.show()


def treiner_NN():
    print '\t----- Iniciando Treinamento do MLP ------\n'
    print 'Carregando Valores na Rede...\n'

    X = np.loadtxt('../res/trainX')
    y = np.loadtxt('../res/trainY')

    testeX = np.loadtxt('../res/testX')
    testeY = np.loadtxt('../res/testY')

    sizeInput = X.shape[1]
    sizeHidden = 20
    sizeOutput = 10
    learning_rate = 0.06
    lambd = 1.0
    epochs = 500
    m = X.shape[0]



    MLP = MultiLayerPerception(sizeInput, sizeHidden, sizeOutput, lambd)

    Treinamento_mlp = TrainerMLP(MLP, X, y, testeX, testeY)

    Treinamento_mlp.gradientDescent(epochs, learning_rate)


    print '\n\tTreinamento Realizado com sucesso !!!'

    raw_input()
    _, _, _, _, h0 = MLP.propagation_Values(X)
    acc = (np.round(h0) == y)
    print'\nPrecisao Treinamento: ', np.mean(acc) * 100

    _, _, _, _, h0 = MLP.propagation_Values(testeX)
    acc = (np.round(h0) == testeY)
    print'\nPrecisao Teste: ', np.mean(acc) * 100


if __name__ == '__main__':
    treiner_NN()
