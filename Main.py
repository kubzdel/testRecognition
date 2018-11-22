from __future__ import division, print_function, absolute_import

import datetime

import tensorflow as tf
import tflearn
from scipy._lib.six import xrange
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import random as rand
from skimage import data as img
from PIL import Image
from numpy import array
import os
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_EXTENSION_LENGHT = 4
TEST_COLUMN_SIZE = 6
SINGLE_ANS = 2
def chunkify(lst, n):
    return [lst[i::n] for i in xrange(n)]

def prepareY(answers):
    Y = []
    for ans in answers:
        vector = []
        if ans == 1:
            vector = [1,0,0,0]
        if ans == 2:
            vector = [0,1,0,0]
        if ans == 3:
            vector = [0,0,1,0]
        if ans == 4:
            vector = [0,0,0,1]
        Y.append(vector)

    return Y



def addAnswers(Y):
    for y in range(0,SINGLE_ANS):
        for i in range(0,TEST_COLUMN_SIZE):
            Y.append([1,0,0,0])
        for i in range(0, TEST_COLUMN_SIZE):
            Y.append([0, 1, 0, 0])
        for i in range(0, TEST_COLUMN_SIZE):
            Y.append([0, 0, 1, 0])
        for i in range(0, TEST_COLUMN_SIZE):
            Y.append([0, 0, 0, 1])
    return Y

images = []
masks = []
k = 10
X = []
Y = []
Xtest = []
Ytest = []
MASK_SIZE = 25  # musi być nieparzysta
PHOTO_SAMPLES = 1000
# ładuj zdjęcia
directory = os.getcwd() + "\samples" + '\\'
for folder in os.listdir("samples"):
    folderConent = os.listdir("samples\\"+folder)
    folderConent = sorted(folderConent)
    for file in folderConent:
        pngfile = img.load(directory + folder+"\\"+file, True)
        pngfile = pngfile/255
        X.append(array(pngfile))
    Y = addAnswers(Y)
plt.imshow(array(X[1]))
#plt.show()
x_size,y_size = X[0].shape

answers = [1,1,2,2,3,2,4,4,1,1,4,2,3,3,3,4,1,2,1,1,4,2,3,2]
X, Y = shuffle(X, Y)

averageAcc = 0

trainSamples = []
testSamples = []

trainMasks = []
testMasks = []

Xtrain = X[:(int(len(X)/5)+1) * 4]
Xtrain = Xtrain.reshape(Xtrain.shape[0],x_size,y_size,1)
Ytrain = Y[:(int(len(X)/5)+1) * 4]
Xtest = X[(int(len(X)/5)+1) * 4:]
Xtest =  Xtest.reshape(Xtest.shape[0],x_size,y_size,1)
Ytest = Y[(int(len(X)/5)+1) * 4:]


img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aumg = ImageAugmentation()
img_aumg.add_random_rotation(3)
network = input_data(shape=[x_size, y_size,1],
                     name='InputData')
#    ,                    data_augmentation=img_aumg,data_preprocessing=img_prep)
print(network)
network = conv_2d(network, 32,5, activation='relu')
# #
network = max_pool_2d(network, 2)
#
network = conv_2d(network, 32, 3, activation='relu')
#
#network = conv_2d(network, 64, 3, activation='relu')
#
network = max_pool_2d(network, 2)

network = fully_connected(network, 128, activation='relu')

network = fully_connected(network, 128, activation='relu')
# network = fully_connected(network, 128, activation='relu')
# network = fully_connected(network, 128, activation='relu')
network = dropout(network, 0.1)

network = fully_connected(network, 4, activation='softmax')

network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001)

model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(Xtrain, Ytrain, n_epoch=1000, shuffle=True, validation_set=(Xtest, Ytest),
          show_metric=True, batch_size=64, snapshot_epoch=False)

accuracy = 0

for toPredict, actual in zip(Xtest, Ytest):
    prediction = model.predict([toPredict])
    predicted_class = np.argmax(prediction)
    actual_class = np.argmax(actual)
    if (predicted_class == actual_class):
        accuracy += 1
accuracy = accuracy / len(Ytest)
averageAcc += accuracy
print("Acucracy: " + str(accuracy))

model.save("docelowy4/" + str(accuracy) + " eye-veins-classifier.tfl")

print("Network trained and saved as eye-veinsclassifier.tfl!")


averageAcc = averageAcc / k
print(averageAcc)
text_file = open("docelowy4/averageACC.txt", "w")
text_file.write("Średnia dokładność: " + str(averageAcc))
text_file.close()
