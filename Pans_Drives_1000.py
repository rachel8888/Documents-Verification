# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:51:16 2020

@author: lenovo
"""


import keras 
import tensorflow as tf
import os
from PIL import Image
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import cv2

path1 = r'C:\Users\lenovo\Desktop\python\Dataset\input_data'
path2 = r'C:\Users\lenovo\Desktop\python\Dataset\Inputed_data'
img_row , img_col = 200 , 200
img_channels = 1

listing = os.listdir(path1)
num_samples = len(listing)
for file in listing:
    im = Image.open(path1 + '\\' + file)  
    img = im.resize((img_row,img_col))
    gray = img.convert('L')
                #need to do some more processing here          
    gray.save(path2 +'\\' +  file, "PNG")

imlist = os.listdir(path2)
im = (Image.open(path2+ '\\' + imlist[0]))
im1 = np.asarray(im)
immatrix = np.asarray([np.asarray(Image.open(path2+ '\\' + im2)).flatten()
              for im2 in imlist],'f')
label=np.ones((num_samples,),dtype = int)

label[0:499]=0
label[499:]=1

data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 3
# number of epochs to train
nb_epoch = 20


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

(X, y) = (train_data[0],train_data[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


X_train = X_train.reshape(X_train.shape[0], 1, img_row, img_col)
X_test = X_test.reshape(X_test.shape[0], 1, img_row, img_col)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')



X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=15, batch_size=64, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

train_loss=history.history['loss']
val_loss=history.history['val_loss']
train_acc=history.history['acc']
val_acc=history.history['val_acc']
xc=range(nb_epoch)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

fname = "weights.hdf5"
model.save_weights(fname)
model.load_weights(fname)

img = cv2.imread(r"C:\Users\lenovo\Desktop\python\Dataset\Pans\Pan0.png",cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (200,200))
img = np.reshape(img, [1,200,200])
pre = np. argmax(model.predict(img))

if np.argmax(pre) == 1 : str_label = 'Pan'
else : str_label = 'Drive'

img = np.reshape(img , [200,200])

plt.imshow(img)
plt.title(str_label)


