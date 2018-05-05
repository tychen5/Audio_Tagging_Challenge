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
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)

from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D, 
                          GlobalMaxPool1D, Input, MaxPool1D, concatenate)

from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation)

from keras import backend as K                      

from sklearn.model_selection import KFold 


# category map dict =====================================================
map_dict = pk.load(open('data/map.pkl' , 'rb'))


X = np.load('data/train/train3_X.npy')
Y_train = pd.read_csv('data/train/train3_Y.csv')


Y_train['trans'] = Y_train['label'].map(map_dict)
Y_train['onehot'] = Y_train['trans'].apply(lambda x: to_categorical(x,num_classes=41))

Y = Y_train['onehot'].tolist()

print('X shape : ')
print(X.shape)

print('Y_train shape : ')
print(Y_train.head(10))

print('Y onehot shape :')
Y = np.array(Y)
Y = Y.reshape(-1 , 41)
print(Y.shape)

#  Normalization =====================================================
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean)/std

# conv2d func =====================================================

def get_2d_conv_model(data):

    nclass = len(Y[0])
    
    # print(data.shape)
    inp = Input(shape=(data.shape))
    # print(inp)

    x = Convolution2D(32, (4,10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(0.001)
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model

# five fold cross validation =====================================================
PREDICTION_FOLDER = "predictions_2d_conv"
MODEL_FOLDER = 'model'

if not os.path.exists(PREDICTION_FOLDER):
    os.mkdir(PREDICTION_FOLDER)

if not os.path.exists(MODEL_FOLDER):
    os.mkdir(MODEL_FOLDER)

if os.path.exists('logs/' + PREDICTION_FOLDER):
    shutil.rmtree('logs/' + PREDICTION_FOLDER)

kf = KFold(n_splits=6)

i = 0
for train_index, test_index in kf.split(X):
    
    # print(train_index)
    # print(test_index)
    # print('\n')
    i +=1 

    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # checkpoint = ModelCheckpoint('model/best_%d.h5'%i, monitor='val_loss', verbose=1, save_best_only=True)
    checkpoint = ModelCheckpoint('model/best_%d.h5'%i, monitor='val_acc', verbose=1, save_best_only=True)

    # early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    early = EarlyStopping(monitor="val_acc", mode="max", patience=10)

    tb = TensorBoard(log_dir='./logs/' + PREDICTION_FOLDER + '/fold_%i'%i, write_graph=True)

    callbacks_list = [checkpoint, early, tb]

    print("#"*50)
    print("Fold: ", i)

    model = get_2d_conv_model(X_train[0])

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), callbacks=callbacks_list, 
                        batch_size=48, epochs=1000)

    # model.load_weights('model/best_%d.h5'%i)


    



print('=========================================================')

'''
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4],[4,5] , [3,3]])
y = np.array([1, 2, 3, 4,5,6])
kf = KFold(n_splits=6)
kf.get_n_splits(X)


for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

'''
