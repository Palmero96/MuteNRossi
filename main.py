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

#esto es para cargar el clasificador de caras de OpenCV
face_cascade = cv.CascadeClassifier('bin/src/haarcascade_frontalface_default.xml')

# Inicializamos los intervalos de la piel
H_LowThreshold = 0
H_HighThreshold = 0
S_LowThreshold = 0
S_HighThreshold = 0
V_LowThreshold = 0
V_HighThreshold = 0

# Open saved SVM model
loaded_model = pickle.load(open("bin/src/model.sav", "rb"))

# 320x320
print('Hello world')

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
    # cv.imshow('frame', base)

    #Quitamos la cara
    noFace = faceRemove(frame, face_cascade)
    #Quitamos el fondo
    BackGround = backgroundRemoval(noFace,bg)
    #Binarizamos la imagen
    imbin = binarization(BackGround, H_LowThreshold, H_HighThreshold, S_LowThreshold, S_HighThreshold, V_LowThreshold, V_HighThreshold)
    #Situamos el Bounding box
    bnd, origbnd = Bounding(imbin,frame)


    # cv.imshow('Foreground',BackGround)
    # cv.imshow('User hand', bnd)
    cv.imshow('Bounding box', origbnd)

    # Pasamos la foto de la mano por el SVM
    bnd = np.float32(bnd)/255.0
    bnd = cv.cvtColor(bnd, cv.COLOR_BGR2GRAY)
    data = hog(bnd)
    data = np.array([data])
    print(loaded_model.predict(data))

    #teclas
    k = cv.waitKey(1) & 0xFF
    if k == ord('q'):#quit
        break
    elif k == ord('b'):#remove background
        print('Obtenemos nuevo background')
        bg = frame
    elif k == ord('s'):#capture skin
        print('Capturamos la piel del usuario')
        th = captureSkin(BackGround)
        # Asign calculated thresholds to used ones
        H_LowThreshold = th[0]
        H_HighThreshold = th[1]
        S_LowThreshold = th[2]
        S_HighThreshold = th[3]
        V_LowThreshold = th[4]
        V_HighThreshold = th[5]


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
