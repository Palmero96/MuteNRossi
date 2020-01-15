import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
from numpy import pi as pi
from numpy.linalg import norm
from _hog import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle

#
#                     HERE BEGINS MAIN FUNCTION
#

# In order to apply HoG we will use OpenCV function
# First we stack the desired samples in a variable
data = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')
# data = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L')
samp = 35
letnum = len(data)
# X = np.zeros(shape=(samp*letnum,15876))
X = np.zeros(shape=(samp*letnum,2916))
Y = list()
num = 0

for l in data:
    path = '../../../../asl_alphabet_train/'+l+'/'+l
    num +=1

    # We'll take a sample of 100 for each letter
    for i in range(1,samp+1):
        dpath = path+str(i)+'.jpg'

        img = cv.imread(dpath)
        if (len(img) != None):
            print(dpath + '   has been succesfully loaded!')

            img = img[8:199-8,8:199-8]
            dim = (160, 160)
            img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
            img = np.float32(img)/255.0
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # cv.imshow('Imagen', img)
            # cv.waitKey(0)

            # The function hog will only accept 1 dimension images (Gray)
            X[(num-1)*samp + i-1] = hog(img)
            Y.append(l)
        else:
            print("Error! no image found")
            i = samp



# Now that we already have the variables well fit the VSM
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, Y_train)

Y_pred = svclassifier.predict(X_test)

print("\n\nCLASSIFIER 1: Linear\n\n")
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
print("Accuracy = ", svclassifier.score(X_test, Y_test))


# svclassifier2 = SVC(kernel='sigmoid')
# svclassifier2.fit(X_train, Y_train)
#
# Y_pred = svclassifier2.predict(X_test)
#
# print("\n\nCLASSIFIER 2: Sigmoid\n\n")
# print(confusion_matrix(Y_test, Y_pred))
# print(classification_report(Y_test, Y_pred))
# print("Accuracy = ", svclassifier2.score(X_test, Y_test))
#
#
# svclassifier3 = SVC(kernel='rbf')
# svclassifier3.fit(X_train, Y_train)
#
# Y_pred = svclassifier3.predict(X_test)
#
# print("\n\nCLASSIFIER 3: Gaussian\n\n")
# print(confusion_matrix(Y_test, Y_pred))
# print(classification_report(Y_test, Y_pred))
# print("Accuracy = ", svclassifier3.score(X_test, Y_test))

# Save the model to disk
filename = 'bin/src/model.sav'
pickle.dump(svclassifier, open(filename, 'wb'))
