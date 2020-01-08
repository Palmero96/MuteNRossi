import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from _hog import hog
from _imgProc import *
from _FileVideoStream import FileVideoStream
import threading
import multiprocessing as mp
from multiprocessing import Queue
#import queue
from imutils.video import FPS
import time

BUF_SIZE = 1
q = Queue(BUF_SIZE)

def getPrediction(q):
    # Open saved SVM model
    loaded_model = pickle.load(open("bin/src/model.sav", "rb"))

    while True:
        if not q.empty():
            # Pasamos la foto de la mano por el SVM
            item = q.get()
            # print("Se ha sacado 1 item, quedan {}".format(q.qsize()))
            item = np.float32(item)/255.0
            item = cv.cvtColor(item, cv.COLOR_BGR2GRAY)
            data = hog(item)
            data = np.array([data])
            print(loaded_model.predict(data))

def Imaq(q):
    #esto es para cargar el clasificador de caras de OpenCV
    face_cascade = cv.CascadeClassifier('bin/src/haarcascade_frontalface_default.xml')
    # Inicializamos los intervalos de la piel
    H_LowThreshold = 0
    H_HighThreshold = 0
    S_LowThreshold = 0
    S_HighThreshold = 0
    V_LowThreshold = 0
    V_HighThreshold = 0

    #Enciendo la camara
    fvs = FileVideoStream(0).start()
    time.sleep(1.0)
    #start FPS timer
    fps = FPS().start()

    # if not cap.isOpened():
    #     print("Cannot open camera")
    #     exit()

    #Capturo el primer fame que sera el fondo inicial
    # ret, bg = cap.read()
    bg = fvs.read()
    fps.update()

    while True:
        # # Capture frame-by-frame
        # ret, frame = cap.read()
        # # if frame is read correctly ret is True
        # if not ret:
        #     print("Can't receive frame (stream end?). Exiting ...")
        #     break
        frame = fvs.read()
        base = fvs.read()

        # Ponemos los rectangulos
        cv.rectangle(base, (125, 255), (205, 295), (0, 255, 0), 2)
        cv.rectangle(base, (125, 300), (205, 345), (0, 255, 0), 2)

        # Ponemos algo de texto
        font = cv.FONT_HERSHEY_SIMPLEX
        # cv.putText(base, 'MutenRossi', (58, 58), font, 2, (255, 255, 255), 1, cv.LINE_AA)


        if not q.full():
            #Quitamos la cara
            noFace = faceRemove(frame, face_cascade)
            #Quitamos el fondo
            BackGround = backgroundRemoval(noFace,bg)
            backaux = BackGround
            #Binarizamos la imagen
            imbin = binarization(BackGround, H_LowThreshold, H_HighThreshold, S_LowThreshold, S_HighThreshold, V_LowThreshold, V_HighThreshold)
            #Situamos el Bounding box
            origbnd, bnd = Bounding(imbin,backaux)

            try:
                cv.imshow('Bounding box', origbnd)
            except:
                print("[ERROR] Empty image to show\n")

            q.put(bnd)
            # print("Existen {} fotos en la cola".format(q.qsize()))

        else:
            #Mostramos la imagen original
            cv.imshow('Bounding box', base)

        #teclas
        k = cv.waitKey(1) & 0xFF
        if k == ord('q'):#quit
            break
        elif k == ord('b'):#remove background
            print('Obtenemos nuevo background')
            bg = frame
        elif k == ord('s'):#capture skin
            print('Capturamos la piel del usuario')
            thold = captureSkin(BackGround)

            # Asign calculated thresholds to used ones
            H_LowThreshold = thold[0]
            H_HighThreshold = thold[1]
            S_LowThreshold = thold[2]
            S_HighThreshold = thold[3]
            V_LowThreshold = thold[4]
            V_HighThreshold = thold[5]

        fps.update()

    # Exiting
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv.destroyAllWindows()
    fvs.stop()


############################################################
#                   INICIO DEL MAIN                        #
############################################################

th = mp.Process(target=getPrediction, args=(q,), daemon=True)
th2 = mp.Process(target=Imaq, args=(q,))

if __name__ == '__main__':
    print("Inicio del programa")

    th.start()
    th2.start()

    # When everything done, release the capture
    th2.join()
