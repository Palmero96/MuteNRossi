import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from _hog import hog
from _imgProc import *

#Instrucciones:
#1º: Colocarse a una distancia de unos 40cm de la camara, con la cabeza hacia la izquierda de la pantalla
#2º: Correr el programa y pulsar la b para que capture un nuevo fondo
#3º: Levantar la mano y que la palma ocupe las dos regiones rectangulares, y luego pulsar s, para que capture la piel
#4º: Enjoy

# Open saved SVM model
loaded_model = pickle.load(open("bin/src/model.sav", "rb"))

#320x320
print('hello world')

#Enciendo la camara
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

#Capturo el primer fame que sera el fondo inicial
ret, bg = cap.read()

#Bucle principal
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    base = frame

    # Ponemos los rectangulos
    cv.rectangle(base, (125, 255), (205, 295), (0, 255, 0), 2)
    cv.rectangle(base, (125, 300), (205, 345), (0, 255, 0), 2)

    # Ponemos algo de texto
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(base, 'MutenRossi', (58, 58), font, 2, (255, 255, 255), 1, cv.LINE_AA)

    # Mostramos la imagen
    cv.imshow('frame', base)

    #Quitamos la cara
    noFace = faceRemove(frame)
    #Quitamos el fondo
    BackGround = backgroundRemoval(noFace,bg)
    #Binarizamos la imagen
    bin = binarization(BackGround, H_LowThreshold, H_HighThreshold, S_LowThreshold, S_HighThreshold, V_LowThreshold, V_HighThreshold)
    #Situamos el Bounding box
    origbnd = Bounding(bin,frame)


    cv.imshow('Foreground',BackGround)
    cv.imshow('user skin', bin)
    cv.imshow('Bounding box', origbnd)
    # bnd = np.float32(bnd)/255.0
    # bnd = cv.cvtColor(bnd, cv.COLOR_BGR2GRAY)
    # data = hog(bnd)
    # data = np.array([data])
    # print(loaded_model.predict(data))

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
