'''
reference :
https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data
'''
import os
import shutil

import numpy as np
import pickle as pk
import pandas as pd

from keras.utils import to_categorical ,Sequence
from keras import losses, models, optimizers

from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception

from keras.models import Sequential
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)

from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D,
                          GlobalMaxPool1D, Input, MaxPool1D, concatenate)

from keras.layers import Conv1D, Conv2D

from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation , MaxPooling2D)

from keras.layers import Activation, LeakyReLU

from keras import backend as K

from sklearn.model_selection import KFold

# gpu usage limit ==============================================================
'''
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# 設定 Keras 使用的 TensorFlow Session
tf.keras.backend.set_session(sess)
'''
# category map dict =====================================================
map_dict = pk.load(open('data/map.pkl' , 'rb'))



# conv2d func =====================================================

def get_2d_conv_model(data):

    nclass = 41
    
    # print(data.shape)
    inp = Input(shape=(data.shape))
    # print(inp)

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5), input_shape=data.shape, padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=0.03))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=0.03))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=0.03))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.35))
    
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=0.03))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(256, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(nclass, activation='softmax', kernel_initializer='glorot_normal'))

    model.summary()
    opt = optimizers.Adam(0.0001)
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model



# five fold cross validation =====================================================
MODEL_FOLDER = 'model_InceptionResNetV2'

if not os.path.exists(MODEL_FOLDER):
    os.mkdir(MODEL_FOLDER)



for i in range(1,11):


    X_train = np.load('data/ten_fold_data/X_train_{}.npy'.format(i)) 
    Y_train = np.load('data/ten_fold_data/Y_train_{}.npy'.format(i)) 
    X_test = np.load('data/ten_fold_data/X_valid_{}.npy'.format(i))
    Y_test = np.load('data/ten_fold_data/Y_valid_{}.npy'.format(i))
    

    checkpoint = ModelCheckpoint('model_InceptionResNetV2/best_%d_{val_acc:.5f}.h5'%i, monitor='val_acc', verbose=1, save_best_only=True)

    early = EarlyStopping(monitor="val_acc", mode="max", patience=10)


    callbacks_list = [checkpoint, early]

    print("#"*50)
    print("Fold: ", i)

    model = get_2d_conv_model(X_train[0])
    
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

   # model.summary()

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), callbacks=callbacks_list,
                        batch_size=128, epochs=10000)
    



