from keras.models import load_model
import numpy as np
import sys
import os
import pickle as pk
import pandas as pd
from os import listdir
from os.path import isfile, join

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


# calculate accuracy
from sklearn.metrics import accuracy_score


# load data 
map_dict = pk.load(open('data/map.pkl' , 'rb'))
reverse_dict = pk.load(open('data/map_reverse.pkl' , 'rb'))
name = pd.read_csv('data/sample_submission.csv')
X_name = name['fname'].tolist()

# normalize
X = np.load("data/X_test.npy")
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean)/std


# load models
mypath = 'cnn2d_verified_refine'
models = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]


# # predict ================================================
csv_folder = 'cnn2d_semi_test_csv'

if not os.path.exists(csv_folder):
    os.mkdir(csv_folder)


for i , m  in enumerate(models):
    i+=1
    print('round : {}'.format(i))

    # predict
    model = load_model(m) 
    result = model.predict(X , verbose = True , batch_size = 256)
    

    df = pd.DataFrame(result)
    df.insert(0,'fname' , X_name)
    df.to_csv('{}/mike_cnn2d_semi_test_{}.csv'.format(csv_folder,i), index=False)

    

