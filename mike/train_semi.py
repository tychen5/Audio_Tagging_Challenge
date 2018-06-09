'''
reference :
https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data
'''
import os
from os import listdir
from os.path import isfile, join
import shutil

import numpy as np
import pickle as pk
import pandas as pd


from keras.utils import to_categorical ,Sequence
from keras import losses, models, optimizers
from keras.models import Sequential
from keras.activations import relu, softmax
from keras.models import load_model
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
from sklearn.utils import shuffle

import resnet

map_dict = pk.load(open('data/map.pkl' , 'rb'))


model_path = 'resnet_varified'
refine_path = 'resnet_varified_refine'

models = [join(model_path, f) for f in listdir(model_path) if isfile(join(model_path, f))]

if not os.path.exists(refine_path):
    os.mkdir(refine_path)



X_train_semi = np.load('data/mfcc/X_train_ens_verified.npy')
df = pd.read_csv('data/mfcc/Y_train_ens_verified.csv')
df['trans'] = df['label_verified'].map(map_dict)
df['onehot'] = df['trans'].apply(lambda x: to_categorical(x,num_classes=41))

Y_train_semi =  df['onehot'].tolist()
Y_train_semi = np.array(Y_train_semi)
Y_train_semi = Y_train_semi.reshape(-1 ,41)


# normalize X_train_semi
mean = np.mean(X_train_semi, axis=0)
std = np.std(X_train_semi, axis=0)
X_train_semi = (X_train_semi - mean)/std


for i , m  in enumerate(models):
    
    # fold 1 - 10  , enumerate 0 - 9
    i += 1
    print(i)
    X_train = np.load('data/ten_fold_data/X_train_{}.npy'.format(i)) 
    Y_train = np.load('data/ten_fold_data/Y_train_{}.npy'.format(i)) 
    X_test = np.load('data/ten_fold_data/X_valid_{}.npy'.format(i))
    Y_test = np.load('data/ten_fold_data/Y_valid_{}.npy'.format(i))

    print('verified data:')
    print(X_train.shape)
    print(Y_train.shape)
    
    
    #append semi data 
    X_train = np.append(X_train,X_train_semi , axis=0)
    Y_train = np.append(Y_train,Y_train_semi , axis=0)
    
    # shuffle new data
    X_train , Y_train = shuffle(X_train, Y_train, random_state=0) 


    print('after append semi data:')
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    model = load_model(m)
    checkpoint = ModelCheckpoint(join(refine_path , 'best_semi_%d_{val_acc:.5f}.h5'%i), monitor='val_acc', verbose=1, save_best_only=True)
    early = EarlyStopping(monitor="val_acc", mode="max", patience=5)

    print("#"*50)
    print("Fold: ", i)

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), callbacks=[checkpoint, early],
                        batch_size=128, epochs=10000)

    

    


#     # early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
#     early = EarlyStopping(monitor="val_acc", mode="max", patience=300)


#     callbacks_list = [checkpoint, early]

#     print("#"*50)
#     print("Fold: ", i)

#     # model = get_2d_conv_model(X_train[0])
#     model = resnet.ResnetBuilder.build_resnet_18((1, 40, 345), 41)
    
#     model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

#    # model.summary()

#     history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), callbacks=callbacks_list,
#                         batch_size=128, epochs=10000)
    