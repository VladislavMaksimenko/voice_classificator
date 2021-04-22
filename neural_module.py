from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image as kerasImage
import numpy as np
import os
import shutil

"""
Полностью повторяем модель, которая использовалась для обучения
"""
def createModel():
    img_width, img_height = 150, 150
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Dense(1, activation='sigmoid', name='output'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

"""
Загружаем сохраненные синапсы
"""
def loadWeights(model, weights):
    model.load_weights(weights)
    return model

"""
Классификация одного изображения
"""
def classifySingleImage(imgSrc, model, threshold):
    img = kerasImage.load_img(imgSrc, target_size=(150, 150))
    x = kerasImage.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x*1./255 
    prediction = model.predict(x)
    if prediction < threshold:
        return True
    return False

