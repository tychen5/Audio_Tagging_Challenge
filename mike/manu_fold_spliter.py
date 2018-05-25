import pandas as pd 
import numpy as np 
from sklearn.model_selection import KFold
import pickle as pk 

from keras.utils import to_categorical ,Sequence

map_dict = pk.load(open('data/map.pkl' , 'rb'))

df = pd.read_csv('data/train_label.csv')  
manu_var = df['manually_verified'].values

# audio_martix = np.load('data/raw/X_train.npy')
audio_martix = np.load('data/mfcc/X_train.npy')

df_manu = df[df['manually_verified'] == 1]
df_manu['trans'] = df_manu['label'].map(map_dict)
df_manu['onehot'] = df_manu['trans'].apply(lambda x: to_categorical(x,num_classes=41))

# manu label 
manu_veri_label = df_manu.label.values

# manu index index 
manu_veri_idx = df_manu.index.values

# manu_train
X =  audio_martix[manu_veri_idx]
Y =  df_manu['onehot'].values

print(manu_veri_label[0:10])
print(manu_veri_idx[0:10])
print(df_manu.head(3))
print(X.shape)
print(Y.shape)


kf = KFold(n_splits=10)

for train_index, test_index in kf.split(manu_veri_idx):
    X_train , X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    print('\n')
    print(X_train.shape)
    print(X_test.shape)
    print('\n')
    print(Y_train.shape)
    print(Y_test.shape)
    break
