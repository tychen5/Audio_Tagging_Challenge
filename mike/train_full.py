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

import resnet


# gpu usage limit ==============================================================
'''
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(sess)
'''
# category map dict =====================================================
map_dict = pk.load(open('data/map.pkl' , 'rb'))


X = np.load('data/train/train3_X.npy')
temp = np.load('data/test/test_X.npy')
X = np.concatenate((X,temp))


Y_train = pd.read_csv('data/train/train3_Y.csv')
temp = pd.read_csv('data/test/test_Y.csv')
Y_train = pd.concat([Y_train, temp])


Y_train['trans'] = Y_train['label'].map(map_dict)
Y_train['onehot'] = Y_train['trans'].apply(lambda x: to_categorical(x,num_classes=41))
Y = Y_train['onehot'].tolist()

print('X shape : ')
print(X.shape)

# print('Y_train shape : ')
# print(Y_train.head(10))

print('Y onehot shape :')
Y = np.array(Y)
Y = Y.reshape(-1 , 41)
print(Y.shape)

#  Normalization =====================================================
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean)/std


# five fold cross validation =====================================================
MODEL_FOLDER = 'model_full'

if not os.path.exists(MODEL_FOLDER):
    os.mkdir(MODEL_FOLDER)


kf = KFold(n_splits=10)

i = 0
for train_index, test_index in kf.split(X):

    i +=1

    X_train = np.load('data/ten_fold_data/X_train_{}.npy'.format(i)) 
    Y_train = np.load('data/ten_fold_data/Y_train_{}.npy'.format(i)) 
    X_test = np.load('data/ten_fold_data/X_valid_{}.npy'.format(i))
    Y_test = np.load('data/ten_fold_data/Y_valid_{}.npy'.format(i))
    
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    # checkpoint = ModelCheckpoint('model/best_%d.h5'%i, monitor='val_loss', verbose=1, save_best_only=True)
    checkpoint = ModelCheckpoint('model_full/best_%d_{val_acc:.5f}.h5'%i, monitor='val_acc', verbose=1, save_best_only=True)

    # early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    early = EarlyStopping(monitor="val_acc", mode="max", patience=30)


    callbacks_list = [checkpoint, early]

    print("#"*50)
    print("Fold: ", i)

    # model = get_2d_conv_model(X_train[0])
    model = resnet.ResnetBuilder.build_resnet_18((1, 40, 345), 41)
    
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

   # model.summary()

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), callbacks=callbacks_list,
                        batch_size=128, epochs=10000)
    

    

