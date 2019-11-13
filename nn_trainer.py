import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
#from keras import Dense
#from keras import Sequential

def hogtransformation(mag, angle):
    array = np.array([0,0,0,0,0,0,0,0,0], dtype=float)

    for m2,a2 in zip(mag,angle):
        for m,a in zip(m2,a2):
            # First we transform angles major than 180 into 180-range
            if a >= 180:
                a -= 180

            if a<=20:
                array[0] += m*(20-a)/(20)
                array[1] += m*(1-(20-a)/20)
            elif a<=40:
                array[1] += m*(40-a)/20
                array[2] += m*(1-(40-a)/20)
            elif a<=60:
                array[2] += m*(60-a)/20
                array[3] += m*(1-(60-a)/20)
            elif a<=80:
                array[3] += m*(80-a)/20
                array[4] += m*(1-(80-a)/20)
            elif a<=100:
                array[4] += m*(100-a)/20
                array[5] += m*(1-(100-a)/20)
            elif a<=120:
                array[5] += m*(120-a)/20
                array[6] += m*(1-(120-a)/20)
            elif a<=140:
                array[6] += m*(140-a)/20
                array[7] += m*(1-(140-a)/20)
            elif a<=160:
                array[7] += m*(160-a)/20
                array[8] += m*(1-(160-a)/20)
            elif a<=180:
                array[8] += m*(180-a)/20
                array[0] += m*(1-(180-a)/20)

    # La funcion devuelve el HoG en formato de array 1x9
    return array

def showhog(scale):
    i=0
    for h in HoG:
        u=0
        for angle in h:
            cv.line(im2, ((i%25)*16+4, (i//25)*16+4), ((i%25)*16+4+int(round(angle*2*math.sin(math.radians(u)))), (i//25)*16+4+int(round(angle*2*math.cos(math.radians(u))))), (0,255,0), 2)
            u+=20
        i+=1


# In order to apply HoG we will use OpenCV function
# First we open any desired image
im = cv.imread('../../../../M15.jpg')
im = np.float32(im)/255.0

im2 = cv.resize(im, None, fx=2, fy=2, interpolation=cv.INTER_LINEAR)
# Making image into grey
im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

# get dimensions of image
dimensions = im.shape

# height, width, number of channels in image
height = im.shape[0]
width = im.shape[1]

cv.imshow('im', im)
cv.waitKey(0)

# Image will be divided into 8x8 portions to get their own HoG and will be appended to a list
im8x8 = list()
for x in range(0,width,8):
    for y in range(0,height,8):
        im8x8.append(im[x:x+7, y:y+7])

mag = list()
angle = list()
for im in im8x8:
    gx = (cv.Sobel(im, cv.CV_32F, 1, 0, ksize=1))
    gy = (cv.Sobel(im, cv.CV_32F, 0, 1, ksize=1))

    # We can find magnitude and direction of the gradients
    ma, ang = cv.cartToPolar(gx, gy, angleInDegrees=True)
    mag.append(ma)
    angle.append(ang)

# Now that we have our HoG matrixes we will transform them into one single array codified in 1x9 (Module and angle %20)

i = 0
for m,a in zip(mag, angle):
    if i==0:
        HoG = hogtransformation(m,a)
    else:
        HoG = np.vstack((HoG, hogtransformation(m,a)))
    i+=1


# A continuacion tendremos que normalizar el HoG

#Intentamos representar el hoG
showhog(2);
cv.imshow('im2', im2)
cv.waitKey(0)
