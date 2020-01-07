import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import math
from _hog import hog

# Open saved SVM model
loaded_model = pickle.load(open("bin/src/model.sav", "rb"))

opts = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing')

dim = (160, 160)
data = np.zeros(shape=(len(opts),3888))
num = 0

for o in opts:
    path = '../../../../asl_alphabet_test/asl_alphabet_test/{}_test.jpg'.format(o)

    img = cv.imread(path)
    img = img[8:199-8,8:199-8]
    img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    img = np.float32(img)/255.0
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    data[num] = hog(img)
    num=num+1


# # Now that we have our model we'll import some images to test the model outside the dataset
# img = cv.imread('../../../../fotoc.png')
# img = img[8:199-8,8:199-8]
# dim = (160, 160)
# img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
# img = np.float32(img)/255.0
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
# # img2 = cv.imread('../../../../fotoj.png')
# # img2 = img2[8:199-8,8:199-8]
# # img2 = cv.resize(img2, dim, interpolation = cv.INTER_AREA)
# # img2 = np.float32(img2)/255.0
# # img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
#
# img2 = cv.imread('../../../../asl_alphabet_test/asl_alphabet_test/nothing_test.jpg')
# img2 = img2[8:199-8,8:199-8]
# img2 = cv.resize(img2, dim, interpolation = cv.INTER_AREA)
# img2 = np.float32(img2)/255.0
# img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# data = hog(img)
# data2 = hog(img2)
# data = np.array([data])
# data2 = np.array([data2])
print("Expected >> ")
print(opts)
print("Resulted >> ")
print(loaded_model.predict(data))
# print(loaded_model.predict(data2))
