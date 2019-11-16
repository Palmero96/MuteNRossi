import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
from numpy import pi as pi
from numpy.linalg import norm

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle

def hog(img):
    cellx = celly = 8
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


#
#                     HERE BEGINS MAIN FUNCTION
#

# In order to apply HoG we will use OpenCV function
# First we stack the desired samples in a variable
samp = 50
letnum = 3
X = np.zeros(shape=(samp*letnum,15876))
Y = np.zeros(samp*letnum)

for l in range(0,3):
    if l==0:
        path = 'A'
        y = np.array([1])
    elif l==1:
        path = 'B'
        y = np.array([2])
    elif l==2:
        path = 'C'
        y = np.array([3])

    path = '../../../../asl_alphabet_train/'+path+'/'+path

    # We'll take a sample of 100 for each letter
    for i in range(1,samp+1):
        dpath = path+str(i)+'.jpg'
        print(dpath + '   has been succesfully loaded!')

        img = cv.imread(dpath)
        img = img[8:199-8,8:199-8]
        img = np.float32(img)/255.0
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # cv.imshow('Imagen', img)
        # cv.waitKey(0)S

        # The function hog will only accept 1 dimension images (Gray)
        X[l*samp + i-1] = hog(img)
        Y[l*samp + i-1] = y

print(Y)
# Now that we already have the variables well fit the VSM
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, Y_train)

Y_pred = svclassifier.predict(X_test)

print("\n\nCLASSIFIER 1: Linear\n\n")
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
print("Accuracy = ", svclassifier.score(X_test, Y_test))


svclassifier2 = SVC(kernel='sigmoid')
svclassifier2.fit(X_train, Y_train)

Y_pred = svclassifier2.predict(X_test)

print("\n\nCLASSIFIER 2: Sigmoid\n\n")
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
print("Accuracy = ", svclassifier2.score(X_test, Y_test))


svclassifier3 = SVC(kernel='rbf')
svclassifier3.fit(X_train, Y_train)

Y_pred = svclassifier3.predict(X_test)

print("\n\nCLASSIFIER 3: Gaussian\n\n")
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
print("Accuracy = ", svclassifier3.score(X_test, Y_test))

# Save the model to disk
filename = 'bin/src/model.sav'
pickle.dump(svclassifier, open(filename, 'wb'))
