import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

H_LowThreshold = 0
H_HighThreshold = 0
S_LowThreshold = 0
S_HighThreshold = 0
V_LowThreshold = 0
V_HighThreshold = 0

def captureSkin(frame):
    #cambiamos a HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # convertimos a arrays todos los canales de las dos areas
    area1H = np.array(hsv[250:270, 140:160, 0])  # recuerda que lo de la izda es Y y lo de la dcha es X
    area1S = np.array(hsv[250:270, 140:160, 1])
    area1V = np.array(hsv[250:270, 140:160, 2])
    area2H = np.array(hsv[300:320, 140:160, 0])
    area2S = np.array(hsv[300:320, 140:160, 1])
    area2V = np.array(hsv[300:320, 140:160, 2])

    area1 = np.array(hsv[250:270, 140:160])
    area2 = np.array(hsv[300:320, 140:160])

    # Hacemos las medias de cada canal
    mean1H = np.mean(area1H)
    mean1S = np.mean(area1S)
    mean1V = np.mean(area1V)
    mean2H = np.mean(area2H)
    mean2S = np.mean(area2S)
    mean2V = np.mean(area2V)

    mean1 = np.mean(area1)
    mean2 = np.mean(area2)

    # estos offset son para ajustar mejor el rango de reconocimiento
    H_offsetLowThreshold = 80
    H_offsetHighThreshold = 30

    S_offsetLowThreshold = 80
    S_offsetHighThreshold = 30

    V_offsetLowThreshold = 80
    V_offsetHighThreshold = 30

    # calculo el Threshold minimo y maximo de cada canal
    global H_LowThreshold
    global H_HighThreshold
    global S_LowThreshold
    global S_HighThreshold
    global V_LowThreshold
    global V_HighThreshold

    H_LowThreshold = min(mean1H, mean2H) - H_offsetLowThreshold
    H_HighThreshold = max(mean1H, mean2H) + H_offsetHighThreshold

    S_LowThreshold = min(mean1S, mean2S) - S_offsetLowThreshold
    S_HighThreshold = max(mean1S, mean2S) + S_offsetHighThreshold

    V_LowThreshold = min(mean1V, mean2V) - V_offsetLowThreshold
    V_HighThreshold = max(mean1V, mean2V) + V_offsetHighThreshold

    return

def binarization(f, H_LowThreshold, H_HighThreshold, S_LowThreshold, S_HighThreshold, V_LowThreshold, V_HighThreshold):
    #Cambiamos a HSV
    hsv = cv.cvtColor(f, cv.COLOR_BGR2HSV)

    #Estos son los rangos entre los que se deberia mover los valores HSV de la piel del usuario
    lower_skin = np.array([H_LowThreshold, S_LowThreshold, V_LowThreshold])
    upper_skin = np.array([H_HighThreshold, S_HighThreshold, V_HighThreshold])

    #Imagen binarizada de la piel

    #Kernel de 3x3
    kernelo = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    kerneld = cv.getStructuringElement(cv.MORPH_DILATE,(3,3))
    #kernelc = cv.getStructuringElement(cv.MORPH_CLOSE,(3,3))

    #Aplicamos los rangos para los canales
    mask = cv.inRange(hsv, lower_skin, upper_skin)

    #AQUI ES DONDE TENEIS QUE PROBAR OPERACIONES MORFOLOGICAS
    #closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernelc)
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernelo)
    #dilation = cv.morphologyEx(mask, cv.MORPH_DILATE, kerneld)
    dilation = cv.dilate(opening, kerneld, iterations=4)

    #Esto es si queremos quedarnos solo con la piel tal cual
    #res = cv.bitwise_and(frame, frame, mask=mask)

    return dilation

def backgroundRemoval(frame, bg):

    #Radio de incertidumbre para aumenat el rango de valores que pertenecen al fondo
    radius = 10

    #Transformamos a escala de grises el fotograma y el fondo
    f = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    bg = cv.cvtColor(bg, cv.COLOR_BGR2GRAY)

    #Vamos pixel a pixel comprobando si caen dentro del rango establecido, si es asi lo ponemos a 0, si no a 255
    for i in range(f.shape[1]):
        for j in range(f.shape[0]):
            if((f.item(j,i)>=(bg.item(j,i)-radius)) and (f.item(j,i)<=(bg.item(j,i)+radius))):
                f.itemset((j,i),0)
            else:
                f.itemset((j,i),255)
    #Aplicamos el filtro de la mediana para eliminar el ruido
    median = cv.medianBlur(f, 5)

    #Aplicamos la mascara al fotograma original
    res = cv.bitwise_and(frame, frame, mask=median)
    return res

print('hello world')

#Enciendo la camara
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

#Capturo el primer fame que sera el fondo inicial
ret, bg = cap.read()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break


    # get dimensions of image
    dimensions = frame.shape

    # height, width, number of channels in image
    height = frame.shape[0]
    width = frame.shape[1]
    channels = frame.shape[2]

    #Quitamos el fondo
    BackGround = backgroundRemoval(frame,bg)

    #binarizamos la imagen
    bin = binarization(BackGround, H_LowThreshold, H_HighThreshold, S_LowThreshold, S_HighThreshold, V_LowThreshold, V_HighThreshold)

    #Ponemos los rectangulos
    cv.rectangle(frame, (135, 245), (165, 275), (0, 255, 0), 2)
    cv.rectangle(frame, (135, 295), (165, 325), (0, 255, 0), 2)

    #Ponemos algo de texto
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame, 'MutenRossi', (58, 58), font, 2, (255, 255, 255), 1, cv.LINE_AA)

    #Mostramos la imagen
    cv.imshow('frame', frame)
    cv.imshow('Foreground',BackGround)
    cv.imshow('user skin', bin)


    #teclas
    k = cv.waitKey(1) & 0xFF
    if k == ord('q'):#quit
        break
    elif k == ord('b'):#remove background
        print('Obtenemos nuevo background')
        bg = frame
    elif k == ord('s'):#capture skin
        print('Capturamos la piel del usuario')
        captureSkin(BackGround)

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()