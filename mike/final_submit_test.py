from keras.models import load_model
import numpy as np
import sys
import os
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

# calculate accuracy
from sklearn.metrics import accuracy_score


def write2CSV(pred, path):
    with open(path, 'w') as f:
        print('id,label', file=f)
        print('\n'.join(['{},{}'.format(i, p) for (i, p) in enumerate(pred)]), file=f)


# load data 
map_dict = pk.load(open('data/map.pkl' , 'rb'))


X = np.load('X_test.npy')
name = pd.read_csv('data/sample_submission.csv')
X_name = name['fname'].tolist()

# print(name.head(3))
# print(X_name[0:3])


# #  Normalization =====================================================
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean)/std

# # predict ================================================
head = list(range(0,41))
head = [str(x) for x in head]


csv_folder = 'predict_csv'
if not os.path.exists(csv_folder):
    os.mkdir(csv_folder)

score = 0.0

for i in range(1,11):
    print('round : {}'.format(i))

    # predict
    '''
    model = load_model('model_full/best_{}.h5'.format(i)) 
    result = model.predict(X)
    np.save('predict_csv/result_{}'.format(i),result)
    '''

    # write to csv 
    '''
    result = np.load('predict_test/mike_predict_{}.npy'.format(i)) 
    df = pd.DataFrame(result , columns = head)
    df.insert(0, 'ID', Y_index)
    df.to_csv('predict_csv/mike_predict_{}.csv'.format(i), index=False)
    '''

    result = np.load('predict_csv/result_{}.npy'.format(i))

    # calculate accuracy
    score += result

# pred = np.argmax(score, axis=-1)
# print(pred[0:3])

# pick best 3 class
output = []
reverse_dict = pk.load(open('data/map_reverse.pkl' , 'rb'))

# print(reverse_dict)

for i , d in enumerate( score):
    # print(i)
    # print(d)
    # print('\n')
    top3 = d.argsort()[-3:][::-1]
    result = [reverse_dict[x] for x in top3]
    s = ' '.join(result)

    output.append(s)
    
    
df = pd.DataFrame(output , columns = ['label'])
df.insert(0,'fname' , X_name)
df.to_csv('ans.csv', index=False)
print(df.head(3))
    







