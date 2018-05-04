'''
reference :
https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data
'''
import os
import shutil

import numpy as np 
import pickle as pk 
import pandas as pd

from keras.utils import to_categorical
from sklearn.model_selection import KFold 


# category map dict =====================================================
map_dict = pk.load(open('data/map.pkl' , 'rb'))


X = np.load('data/train/train3_X.npy')
Y_train = pd.read_csv('data/train/train3_Y.csv')


Y_train['trans'] = Y_train['label'].map(map_dict)
Y_train['onehot'] = Y_train['trans'].apply(lambda x: to_categorical(x,num_classes=41))

Y = Y_train['onehot'].values

print('X shape : ')
print(X.shape)

print('Y_train shape : ')
print(Y_train.head(10))

print('Y onehot shape :')
print(Y.shape)

#  Normalization =====================================================
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X = (X - mean)/std


# five fold cross validation =====================================================
PREDICTION_FOLDER = "predictions_2d_conv"
if not os.path.exists(PREDICTION_FOLDER):
    os.mkdir(PREDICTION_FOLDER)

kf = KFold(n_splits=6)

for train_index, test_index in kf.split(X):
    print(train_index)
    print(test_index)
    print('\n')
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
