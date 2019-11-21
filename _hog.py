import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
from numpy import pi as pi
from numpy.linalg import norm

def hog(img):
    cellx = celly = 32
    bin_n = 9 #Number of bins
    bin = np.int32(180/bin_n)

    n_cellx = int(img.shape[0]/cellx)
    n_celly = int(img.shape[1]/celly)

    hist = np.zeros(shape=(n_cellx, n_celly, bin_n))
    for i in range(0, n_cellx):
        histy = np.zeros(shape=(n_celly, bin_n))
        for j in range(0, n_celly):
            gx = cv.Sobel(img[i*celly:(i+1)*celly, j*cellx:(j+1)*cellx], cv.CV_32F, 1, 0)
            gy = cv.Sobel(img[i*celly:(i+1)*celly, j*cellx:(j+1)*cellx], cv.CV_32F, 0, 1)

            m,a = cv.cartToPolar(gx, gy, angleInDegrees=True)

            v = np.zeros(bin_n, dtype=float)
            for rawm2, rawa2 in zip(m, a):
                for rawm, rawa in zip(rawm2, rawa2):
                    if rawa >= 180:
                        rawa = rawa - 180
                        if rawa==180:
                            rawa = 0
                    index = int(rawa//bin)
                    v[index] += rawm*(rawa - index*bin)/bin
                    if index == bin_n-1:
                        v[0] += rawm*(180 - rawa)/bin
                    else:
                        v[index + 1] += rawm*(index*bin + bin - rawa/bin)
            histy[j] = v
        hist[i] = histy[j]

    # The array hist with dimension 3 represents the HoG for each cell (x,y)
    # Now we will normalize de histogram
    norm_hist = np.array([])
    for j in range(0, n_celly-1):
        for i in range(0, n_cellx-1):
            aux = hist[i,j]
            aux = np.concatenate((aux, hist[i+1,j]))
            aux = np.concatenate((aux, hist[i,j+1]))
            aux = np.concatenate((aux, hist[i+1,j+1]))

            aux_n = 0
            for a in aux:
                aux_n += a**2
            aux_n = math.sqrt(aux_n)

            if aux_n != 0:
                norm_hist = np.concatenate((norm_hist, aux/aux_n))
            else:
                norm_hist = np.concatenate((norm_hist, aux))

    # Norm_hist is a large array containing all the pixel's hogs concatenated
    return norm_hist
