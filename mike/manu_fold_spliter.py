import pandas as pd 
import numpy as np 
from sklearn.model_selection import KFold
import pickle as pk 
import os

from keras.utils import to_categorical ,Sequence
pd.options.mode.chained_assignment = None  # default='warn'

map_dict = pk.load(open('data/map.pkl' , 'rb'))

df = pd.read_csv('data/train_label.csv')  

# audio_martix = np.load('data/raw/X_train.npy')
audio_martix = np.load('data/mfcc/X_train.npy')

# df_unmanu ================================================
df_manu = df[df['manually_verified'] == 1]
df_manu['trans'] = df_manu['label'].map(map_dict)
df_manu['onehot'] = df_manu['trans'].apply(lambda x: to_categorical(x,num_classes=41))

# manu index index 
manu_veri_idx = df_manu.index.values

# manu_train
X =  audio_martix[manu_veri_idx]
Y =  df_manu['onehot'].tolist()
Y = np.array(Y)
Y = Y.reshape(-1 ,41)

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean)/std

print(manu_veri_idx[0:10])
print(df_manu.head(3))
print(X.shape)
print(Y.shape)

fold_path = 'data/ten_fold_data'
if not os.path.exists(fold_path):
    os.mkdir(fold_path)


kf = KFold(n_splits=10)
k = 0
# split manu data
for train_index, test_index in kf.split(manu_veri_idx):
    k+=1
    X_train , X_valid = X[train_index], X[test_index]
    Y_train, Y_valid = Y[train_index], Y[test_index]
    
    np.save( os.path.join(fold_path, 'X_train_{}'.format(k)), X_train)
    np.save( os.path.join(fold_path, 'Y_train_{}'.format(k)), Y_train)
    np.save( os.path.join(fold_path, 'X_valid_{}'.format(k)), X_valid)
    np.save( os.path.join(fold_path, 'Y_valid_{}'.format(k)), Y_valid)
    print('{} fold split done'.format(k))

print('verified  data split done =====================')
# df_unmanu ================================================

df_unmanu = df[df['manually_verified'] == 0]
df_unmanu['trans'] = df_unmanu['label'].map(map_dict)
df_unmanu['onehot'] = df_unmanu['trans'].apply(lambda x: to_categorical(x,num_classes=41))
unmanu_veri_idx = df_unmanu.index.values
X = audio_martix[unmanu_veri_idx]
Y = df_unmanu['onehot'].tolist()
Y = np.array(Y)
Y = Y.reshape(-1 ,41)

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean)/std

np.save( os.path.join(fold_path, 'X_unverified'.format(k)), X)
np.save( os.path.join(fold_path, 'Y_unverified'.format(k)), Y)


print(unmanu_veri_idx[0:10])
print(df_unmanu.head(3))
print(X.shape)
print(Y.shape)
print('unverified data done =====================')