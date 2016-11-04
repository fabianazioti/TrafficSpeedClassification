import cv2
import numpy as np
import cv2.cv as cv
#from classificarSin import ClassficacaoNum
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
        'Hmin2' : 160,
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

def verificaTamanhoNumero(candidato):

    width, heigth = candidato[1]
    if(heigth == 0 or width == 0):
        return False
    if(width >= 58 or heigth >= 60):
        return False
    if( width <= 10 or heigth <= 15):
        if(heigth >= 20):
            True
        else:
            return False


def getKey(item):
    return item[0][0]

class Placa():
    '''
    classdocs
    '''
    def __init__(self, img, posicao_x, posicao_y,raio):
        '''
        Constructor
        '''
        self.img = img
        self.img_original = self.img.copy()
        self.posicao_x = posicao_x
        self.posicao_y = posicao_y
        self.raio = raio
        self.contornosIMG()

        self.classifica()


    def __repr__(self):
        return 'Placa({0.posicao_x!r}, {0.posicao_y!r}, {0.raio!r})'.format(self)

    def __str__(self):
        return '({0.posicao_x!s}, {0.posicao_y!s}, {0.raio!s})'.format(self)

    def aplicaOperacoesIMG(self):
        # ------- Transforma a imagem em escala de cinza
        self.img = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)

        self.img = cv2.equalizeHist(self.img)

        # --- Aplica um blur e transforma em binario
        blur = cv2.GaussianBlur(self.img,(5,5),0)
        _, self.img = cv2.threshold(blur, 0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)



        #operacao morfologica
        kernel = np.ones((3,3),np.uint8)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel)

        # plt.subplot(),plt.imshow(self.img, cmap='gray'),plt.title('Abertura')
        # plt.show()

    def contornosIMG(self):
        self.aplicaOperacoesIMG()

        self.contornos = []
        #determina contornos dentro da placa
        contours,_ = cv2.findContours(self.img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            if(verificaTamanhoNumero(rect) == False):
                pass
            else:
                xInicial,yInicial,w,h = cv2.boundingRect(cnt)
                xFinal = (xInicial + w) + 2
                yFinal = (yInicial + h) + 2

                if(self.verificaPosicaoNumero(xInicial, xFinal, yInicial, yFinal) == False):
                    pass
                else:
                    global TOKEN
                    TOKEN = True

                    self.contornos.append(((xInicial,yInicial),(xFinal,yFinal)))
                    cv2.rectangle(self.img_original,(xInicial,yInicial),(xFinal,yFinal),(0,255,0),2)

#         imgRGB = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)
#         plt.subplot(),plt.imshow(imgRGB),plt.title('Numeros Detectados')
#         plt.show()

    def verificaPosicaoNumero(self, xInicial, xFinal, yInicial, yFinal):
        x_OInicial = 2
        y_OInicial = 2
        x_OFinal, y_OFinal,_ = self.img_original.shape

        x_OFinal += -1
        y_OFinal += -3
        if xInicial >= x_OInicial and xFinal <= x_OFinal:
            if yInicial > y_OInicial and yFinal < y_OFinal:
                return True
        else:
            print 'Posicao errada'
            return False

    def classifica(self):
        print('Classificando')
        num = len(self.contornos) -1
        self.contornos.sort(key=getKey)
        self.valor = 0

        for cnt in self.contornos:

            numberRoi = self.img_original[cnt[0][1]:cnt[1][1],cnt[0][0]:cnt[1][0]]
            numero = ClassficacaoNum(numberRoi)

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
        self.Imagem = cv2.medianBlur(self.Imagem,5)
        self.Imagem = cv2.cvtColor(self.Imagem, cv2.COLOR_BGR2HSV)

        menor_vermelho = np.array([HSV['Hmin1'], HSV['Smin1'], HSV['Vmin1']])
        maior_vermelho = np.array([ HSV['Hmax1'], HSV['Smax1'], HSV['Vmax1']])

        menor2_vermelho = np.array([ HSV['Hmin2'], HSV['Smin2'], HSV['Vmin2']])
        maior2_vermelho = np.array([ HSV['Hmax2'], HSV['Smax2'], HSV['Vmax2']])

        mask = cv2.inRange(self.Imagem, menor_vermelho, maior_vermelho)
        mask2 = cv2.inRange(self.Imagem, menor2_vermelho, maior2_vermelho)
        self.Imagem = cv2.addWeighted(mask,1.0,mask2,1.0,0)

        kernel = np.ones((3,3),np.uint8)
        self.Imagem= cv2.morphologyEx(self.Imagem, cv2.MORPH_OPEN, kernel)
        #plt.subplot(),plt.imshow(self.Imagem, cmap='gray'),plt.title('Segmentacao')
        #plt.show()

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


    def regioesInteresse(self):

        for (xCentro,yCentro,raio) in self.Posicao_circulos[0,:]:
            raio += 3
            yInicial = (yCentro-raio)
            yFinal = (yCentro+raio)
            xInicial = (xCentro-raio)
            xFinal = (xCentro + raio)

            # cv2.circle(self.Imagem_original,(xCentro,yCentro),raio,(0,255,0),2)

            self.Regioes.append(Placa(self.Imagem_original[yInicial:yFinal,xInicial:xFinal],xCentro,yCentro,raio))

        # imgRGB = cv2.cvtColor(self.Imagem_original, cv2.COLOR_BGR2RGB)
        # plt.subplot(),plt.imshow(imgRGB),plt.title('Circulos Detectados')
        # plt.show()

    def desenhaIMG(self):

        for r in self.Regioes:
            center = (int(r.posicao_x),int(r.posicao_y))
            radius = int(r.raio)

            cv2.circle(self.Imagem_original,center,radius,(0,255,0),3)

            cv2.putText(self.Imagem_original, str(r.valor),(r.posicao_x- r.raio, r.posicao_y- r.raio), FONTE, 5,(0,255,0),5,cv2.CV_AA)
        imgRGB = cv2.cvtColor(self.Imagem_original, cv2.COLOR_BGR2RGB)
        plt.subplot(),plt.imshow(imgRGB),plt.title('Numeros Detectados')
        plt.show()

    def desenhaVideo(self):
        if Placa.token == True:
            for r in self.Regioes:
                center = (int(r.posicao_x),int(r.posicao_y))
                radius = int(r.raio)

                cv2.circle(self.Imagem_original,center,radius,(0,255,0),3)
                print r.valor
                cv2.putText(self.Imagem_original, str(r.valor),(r.posicao_x- r.raio, r.posicao_y- r.raio), FONTE, 5,(0,255,0),5,cv2.CV_AA)

            cv2.imshow('Video', cv2.pyrDown(self.Imagem_original))
            return self.Imagem_original
