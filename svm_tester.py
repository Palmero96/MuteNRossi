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

# Now that we have our model we'll import some images to test the model outside the dataset
img = cv.imread('../../../../fotoc.png')
img = img[8:199-8,8:199-8]
img = np.float32(img)/255.0
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img2 = cv.imread('../../../../fotoj.png')
img2 = img2[8:199-8,8:199-8]
img2 = np.float32(img2)/255.0
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

data = hog(img)
data2 = hog(img2)
data = np.array([data])
data2 = np.array([data2])
print(loaded_model.predict(data))
print(loaded_model.predict(data2))
