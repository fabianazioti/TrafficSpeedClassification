import cv2
import numpy as np
import cv2.cv as cv
from MCS import ClassficacaoContorno
from matplotlib import pyplot as plt
from math import pow

TOKEN = False
FONTE = cv2.FONT_HERSHEY_PLAIN

# -------- Padrao de Cor para Imagens Google Maps --------- #
HSV = {
        'Hmin1' : 0,
        'Hmax1' : 10,

        'Smin1' : 50,
        'Smax1' : 255,

        'Vmin1' : 50,
        'Vmax1' : 255,

        'Hmin2' : 150, # 160
        'Hmax2' : 179,

        'Smin2' : 50,
        'Smax2' : 255,

        'Vmin2' : 50,
        'Vmax2' : 255

}

# ------- Padrao de Cor para o Raspberry ------- #

# HSV = {
#         'Hmin1' : 0,
#         'Hmax1' : 15,

#         'Smin1' : 50,
#         'Smax1' : 255,

#         'Vmin1' : 50,
#         'Vmax1' : 255,

#         'Hmin2' : 140,
#         'Hmax2' : 179,

#         'Smin2' : 50,
#         'Smax2' : 255,

#         'Vmin2' : 50,
#         'Vmax2' : 255

# }

# -------- --------- ----- #

def verificar_Tamanho_Numero(candidato):

    largura, altura = candidato[1]
    if(altura == 0 or largura == 0):
        return False
    if(largura >= 58 or altura >= 60):
        return False
    if( largura <= 10 or altura <= 15):
        if(altura >= 30):
            True
        else:
            return False


def getKey(item):
    return item[0][0]

class Placa():
    '''
    Regioes de Interesse - Encontrar os ROI Contornos Numericos
    '''
    def __init__(self, img, posicao_x, posicao_y,raio):
        '''
        Construtor da Classe
        '''
        self.Img = img
        self.Img_original = self.Img.copy()
        self.Posicao_x = posicao_x
        self.Posicao_y = posicao_y
        self.Raio = raio
        self.contornos_Imagem()

        self.classifica()


    def __repr__(self):
        return 'Placa({0.Posicao_x!r}, {0.Posicao_y!r}, {0.Raio!r})'.format(self)

    def __str__(self):
        return '({0.Posicao_x!s}, {0.Posicao_y!s}, {0.Raio!s})'.format(self)

    def processar_Imagem(self):

        # ------- Transformacao da Imagem para binario (utilizando a binarizacao otima) --- #
        self.Img = cv2.cvtColor(self.Img,cv2.COLOR_BGR2GRAY)

        # self.Img = cv2.equalizeHist(self.Img)
        blur = cv2.GaussianBlur(self.Img,(5,5),0)
        _, self.Img = cv2.threshold(blur, 0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)

        kernel = np.ones((4,3),np.uint8)
        self.Img = cv2.morphologyEx(self.Img, cv2.MORPH_OPEN, kernel, iterations =1)

        plt.subplot(),plt.imshow(self.Img, cmap='gray'),plt.title('Numeros')
        plt.show()

    def contornos_Imagem(self):
        self.processar_Imagem()

        self.contornos = []

        # -- Determina os contornos dentro da placa ---- #

        contournos,_ = cv2.findContours(self.Img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contournos:
            rect = cv2.minAreaRect(cnt)
            if(verificar_Tamanho_Numero(rect) == False):
                pass
            else:
                xInicial,yInicial,w,h = cv2.boundingRect(cnt)
                xFinal = (xInicial + w) + 1
                yFinal = (yInicial + h) + 1

                if(self.verificar_Posicao_Numero(xInicial, xFinal, yInicial, yFinal) == False):
                    pass
                else:
                    global TOKEN
                    TOKEN = True

                    self.contornos.append(((xInicial,yInicial),(xFinal,yFinal)))
                    #cv2.rectangle(self.img_original,(xInicial,yInicial),(xFinal,yFinal),(0,255,0),2)

#         imgRGB = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)
#         plt.subplot(),plt.imshow(imgRGB),plt.title('Numeros Detectados')
#         plt.show()

    def verificar_Posicao_Numero(self, xInicial, xFinal, yInicial, yFinal):
        x_OInicial = 2
        y_OInicial = 2
        x_OFinal, y_OFinal,_ = self.Img_original.shape

        x_OFinal += -1
        y_OFinal += -3
        if xInicial >= x_OInicial and xFinal <= x_OFinal:
            if yInicial > y_OInicial and yFinal < y_OFinal:
                return True
        else:
            return False

    def classifica(self):
        num = len(self.contornos) -1
        self.contornos.sort(key=getKey)
        self.valor = 0

        for cnt in self.contornos:

            ROI = self.Img_original[cnt[0][1]:cnt[1][1],cnt[0][0]:cnt[1][0]]
            numero = ClassficacaoContorno(ROI)

            maxX = numero.classificarN()
            self.valor = int(self.valor + maxX * pow(10,num))
            num = num - 1


class DetectarPlacas(object):
    '''
        Classe responsavel por DetectarPlacas sinalizacao de velocidade nas Imagens
    '''
    def __init__(self, img):
        '''
        Constructor da Classe
        '''
        self.Imagem = img
        self.Imagem_original = self.Imagem.copy()
        self.Posicao_circulos = None
        self.Regioes = []

    def segmentacao_Cor(self):
        # -- Realiza a Segmenta da Cor vermelha -- #

        self.Imagem = cv2.medianBlur(self.Imagem, 5)
        self.Imagem = cv2.cvtColor(self.Imagem, cv2.COLOR_BGR2HSV)

        menor_vermelho = np.array([HSV['Hmin1'], HSV['Smin1'], HSV['Vmin1']])
        maior_vermelho = np.array([ HSV['Hmax1'], HSV['Smax1'], HSV['Vmax1']])

        menor2_vermelho = np.array([ HSV['Hmin2'], HSV['Smin2'], HSV['Vmin2']])
        maior2_vermelho = np.array([ HSV['Hmax2'], HSV['Smax2'], HSV['Vmax2']])

        mask = cv2.inRange(self.Imagem, menor_vermelho, maior_vermelho)
        mask2 = cv2.inRange(self.Imagem, menor2_vermelho, maior2_vermelho)
        self.Imagem = cv2.addWeighted(mask,1.0,mask2,1.0,1)

        kernel = np.ones((4,3),np.uint8)
        self.Imagem= cv2.morphologyEx(self.Imagem, cv2.MORPH_OPEN, kernel, iterations = 1)


        plt.subplot(),plt.imshow(self.Imagem, cmap='gray'),plt.title('Segmentacao apos')
        plt.show()

    def detectar_Circulo(self):
        # -- Funcao responsavel por detectar circulos --- #
        xyr_circulos = cv2.HoughCircles(self.Imagem,cv.CV_HOUGH_GRADIENT,1,50,param1=100,param2=20,minRadius=10,maxRadius=80)

        # -- Percorre todos os circulos detectados e retorna suas posicoes -- #
        if xyr_circulos is None:
            print'        # ---- Circulo NAO detectado! ---- #  '
            return None
        else:
            print '        # --- Circulo DETECTADO! --- #'
            self.Posicao_circulos = np.uint16(np.around(xyr_circulos))


    def regioes_Interesse(self):

        for (xCentro,yCentro,raio) in self.Posicao_circulos[0,:]:
            raio += 2
            yInicial = (yCentro-raio)
            yFinal = (yCentro+raio)
            xInicial = (xCentro-raio)
            xFinal = (xCentro + raio)

            self.Regioes.append(Placa(self.Imagem_original[yInicial:yFinal,xInicial:xFinal],xCentro,yCentro,raio))

    def desenha_Resultado_IMG(self):

        for r in self.Regioes:
            center = (int(r.Posicao_x),int(r.Posicao_y))
            radius = int(r.Raio)

            cv2.circle(self.Imagem_original, center,radius,(0,255,0),3)

            cv2.putText(self.Imagem_original, str(r.valor),(r.Posicao_x- r.Raio, r.Posicao_y- r.Raio), FONTE, 5,(0,255,0),5,cv2.CV_AA)
        imgRGB = cv2.cvtColor(self.Imagem_original, cv2.COLOR_BGR2RGB)
        plt.subplot(),plt.imshow(imgRGB),plt.title('Numeros Detectados')
        plt.show()

    def desenha_Resultado_Video(self):
        if Placa.token == True:
            for r in self.Regioes:
                center = (int(r.posicao_x),int(r.posicao_y))
                radius = int(r.raio)

                cv2.circle(self.Imagem_original,center,radius,(0,255,0),3)
                print r.valor
                cv2.putText(self.Imagem_original, str(r.valor),(r.posicao_x- r.raio, r.posicao_y- r.raio), FONTE, 5,(0,255,0),5,cv2.CV_AA)

            cv2.imshow('Video', cv2.pyrDown(self.Imagem_original))
            return self.Imagem_original
