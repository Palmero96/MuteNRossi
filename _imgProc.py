import numpy as np
import cv2 as cv

#funcion que pone los intervalos a los canales H, S y V en función de la piel del usuario
#No devuelve nada, guarda los valores globalmente
def captureSkin(frame):

    #cambiamos a HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # convertimos a arrays todos los canales de las dos areas
    area1H = np.array(hsv[255:295, 125:255, 0])  # recuerda que lo de la izda es Y y lo de la dcha es X (125, 300), (205, 345)
    area1S = np.array(hsv[255:295, 125:255, 1])
    area1V = np.array(hsv[255:295, 125:255, 2])
    area2H = np.array(hsv[300:345, 125:255, 0])
    area2S = np.array(hsv[300:345, 125:255, 1])
    area2V = np.array(hsv[300:345, 125:255, 2])

    area1 = np.array(hsv[255:295, 125:255])
    area2 = np.array(hsv[300:345, 125:255])

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
    H_offsetLowThreshold = 30#80
    H_offsetHighThreshold = 30#30

    S_offsetLowThreshold = 30#80
    S_offsetHighThreshold = 30#30

    V_offsetLowThreshold = 30#80
    V_offsetHighThreshold = 30#30

    Threshold = [0,0,0,0,0,0]

    Threshold[0] = min(mean1H, mean2H) - H_offsetLowThreshold
    Threshold[1] = max(mean1H, mean2H) + H_offsetHighThreshold

    Threshold[2] = min(mean1S, mean2S) - S_offsetLowThreshold
    Threshold[3] = max(mean1S, mean2S) + S_offsetHighThreshold

    Threshold[4] = min(mean1V, mean2V) - V_offsetLowThreshold
    Threshold[5] = max(mean1V, mean2V) + V_offsetHighThreshold

    return Threshold

#Esta funcion toma la imagen sin fondo y la binarza, poniendo a blanco la mano
#Tambien aplica unas operaciones morfologicas para que se vea mas delineada y entera
#Recibe la imagen normal sin fondo, y los 3 pares de limites de los canales H, S, V
#Devuelve la imagen binarizada

def binarization(f, H_LowThreshold, H_HighThreshold, S_LowThreshold, S_HighThreshold, V_LowThreshold, V_HighThreshold):
    #Cambiamos a HSV
    hsv = cv.cvtColor(f, cv.COLOR_BGR2HSV)

    #Estos son los rangos entre los que se deberia mover los valores HSV de la piel del usuario
    lower_skin = np.array([H_LowThreshold, S_LowThreshold, V_LowThreshold])
    upper_skin = np.array([H_HighThreshold, S_HighThreshold, V_HighThreshold])

    #Imagen binarizada de la piel

    #Kernel de 3x3
    kernel = np.ones((3,3),np.uint8)
    kernelo = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    kerneld = cv.getStructuringElement(cv.MORPH_DILATE,(3,3))
    kernelc = np.ones((10, 10), np.uint8)

    #Aplicamos los rangos para los canales

    #aplicamos la mascara para los valores que caigan dentro del rango
    mask = cv.inRange(hsv, lower_skin, upper_skin)
    #aplicamos una apertura
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernelo)
    #aplicamos una dilatacion
    mask = cv.dilate(mask, kernel, iterations=4)
    #aplicamos un cierre
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernelc)

    return mask

#esta funcion quita el fondo en funcion de lo que tiene movimiento o no,
#comparando el siguiente frame con el anterior.
#Los parametros son el frame actual y el frame que este establecido como fondo
def backgroundRemoval(frame, bg):

    #Radio de incertidumbre para aumentar el rango de valores que pertenecen al fondo
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
    median = cv.medianBlur(f, 7)
    #Aplicamos filtro gaussiano
    cv.GaussianBlur(median, (7, 7), 100)
    kernelc = np.ones((16, 16), np.uint8)
    #Aplicamos un cierre
    median = cv.morphologyEx(median, cv.MORPH_CLOSE, kernelc)

    #Aplicamos la mascara al fotograma original
    res = cv.bitwise_and(frame, frame, mask=median)
    return res

#Para evitar que la cara caiga dentro de el rango de valores de los canales,
#se aplica esta funcion para localizar la cara y ponerle un rectangulo negro encima
def faceRemove(f, face_c):

    # Convert to grayscale
    gray = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_c.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv.rectangle(f, (x-20, y-50), (x + w+20, y + h+80), (0, 0, 0), -1)

    return f

#Esta funcion pone el cuadrado sobre la mano y la escala a un formato
#que le guste a Alvaro
#recibe la imagen binarizada y la imagen base
#y devuelve la imagen en el formato que le guste a Alvaro y cuadrada
def Bounding(binar,b):
    contours, hierarchy = cv.findContours(binar,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cnt = max(contours, key = lambda x: cv.contourArea(x))
    else:
        return
    M = cv.moments(cnt)

    #Centroid
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    #cv.rectangle(b, (cx-100, cy-100), (cx+130, cy+130), (255, 0, 0), 2)
    x, y, w, h = cv.boundingRect(cnt)
    mayor = 0
    borde = 10
    # width, height = cv.GetSize(b)
    #me quedo con la dimension del rectangulo mayor
    if(w>h):
        mayor = w
    else:
        mayor = h
        #pongo el rectangulo

    # if x <= borde:
    #     xlow = 1
    # else:
    #     xlow = x-borde
    # if x + borde > width:
    #     xhigh = width
    # else:
    #     xhigh = x+borde
    # if y <= borde:
    #     ylow = 1
    # else:
    #     ylow = y-borde
    # if y + borde > height:
    #     yhigh = height
    # else:
    #     yhigh = y+borde


    # cv.rectangle(b, (xlow, ylow), (xhigh, yhigh), (0, 255, 0), 2)
    cv.rectangle(b, (x,y), (x+mayor, y+mayor), (0,255,0), 2)
    #me quedo con la mano solo
    squared = b[x:x+mayor,y:y+mayor]
    #la cambio de tamanio
    resized = cv.resize(squared,(160,160),interpolation=cv.INTER_AREA)
    return resized, b
