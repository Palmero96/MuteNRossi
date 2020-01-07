import numpy as np
import cv2 as cv

#Instrucciones:
#1º: Colocarse a una distancia de unos 40cm de la camara, con la cabeza hacia la izquierda de la pantalla
#2º: Correr el programa y pulsar la b para que capture un nuevo fondo
#3º: Levantar la mano y que la palma ocupe las dos regiones rectangulares, y luego pulsar s, para que capture la piel
#4º: Enjoy

#esto es para cargar el clasificador de caras de OpenCV
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

#Inicializamos los intervalos de la piel
H_LowThreshold = 0
H_HighThreshold = 0
S_LowThreshold = 0
S_HighThreshold = 0
V_LowThreshold = 0
V_HighThreshold = 0

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
def faceRemove(f):

    # Convert to grayscale
    gray = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv.rectangle(f, (x-20, y-50), (x + w+20, y + h+80), (0, 0, 0), -1)

    return f

#Esta funcion pone el cuadrado sobre la mano y la escala a un formato
#que le guste a Alvaro
#recibe la imagen binarizada y la imagen base
#y devuelve la imagen en el formato que le guste a Alvaro y cuadrada
def Bounding(binar,f):

    try:
        contours, hierarchy = cv.findContours(binar,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key = lambda x: cv.contourArea(x))

        M = cv.moments(cnt)

        #Centroid
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        #cv.rectangle(b, (cx-100, cy-100), (cx+130, cy+130), (255, 0, 0), 2)
        x, y, w, h = cv.boundingRect(cnt)
        mayor = 0
        #me quedo con la dimension del rectangulo mayor
        if(w>h):
            mayor = w
        else:
            mayor = h
            #pongo el rectangulo
        #Bounding rectangle
        X_ini = x-15
        Y_ini = y-15
        X_end = x + mayor + 15
        Y_end = y + mayor + 15
        cv.rectangle(f, (X_ini, Y_ini), (X_end, Y_end), (0, 255, 0), 2)

        # Centroid rectangle
        #X_ini = cx - round(mayor/2)
        #Y_ini = cy - round(mayor/2) - 20
        #X_end = cx + round(mayor/2)
        #Y_end = cy + round(mayor/2)
        #cv.rectangle(f, (X_ini, Y_ini), (X_end, Y_end), (0, 0, 255), 2)

        #me quedo con la mano solo
        #squared = f[X_ini:X_end, Y_ini:Y_end]
        squared = f[Y_ini:Y_end, X_ini:X_end]
        # la cambio de tamanio
        resized = cv.resize(squared, (320, 320), interpolation=cv.INTER_AREA)
        return f, resized
    except:
        print('Nope')
        return f, None

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
    ret, base = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break


    # get dimensions of image
    #dimensions = frame.shape

    # height, width, number of channels in image
    #height = frame.shape[0]
    #width = frame.shape[1]
    #channels = frame.shape[2]


    # Ponemos los rectangulos
    cv.rectangle(base, (125, 255), (205, 295), (0, 255, 0), 2)
    cv.rectangle(base, (125, 300), (205, 345), (0, 255, 0), 2)

    # Ponemos algo de texto
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(base, 'MutenRossi', (58, 58), font, 2, (255, 255, 255), 1, cv.LINE_AA)


    #Quitamos la cara
    noFace = faceRemove(frame)

    #Quitamos el fondo
    BackGround = backgroundRemoval(noFace,bg)

    #Binarizamos la imagen
    bin = binarization(BackGround, H_LowThreshold, H_HighThreshold, S_LowThreshold, S_HighThreshold, V_LowThreshold, V_HighThreshold)

    #Situamos el Bounding box
    bnd,bnd_res = Bounding(bin,frame)

    #print('huauauauauauuaa',bnd_res.shape[0])
    #print('huauauauauauuaa',bnd_res.shape[1])

    try:
        # Mostramos la imagen
        cv.imshow('frame', base)
        cv.imshow('Foreground',BackGround)
        cv.imshow('user skin', bin)
        cv.imshow('Bounding box', bnd_res)
    except:
        print('empty image to show')


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
    elif k == ord('f'):#capture image
        print('Guardamos la imagen resized')
        cv.imwrite('mano.png', bnd_res)

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()