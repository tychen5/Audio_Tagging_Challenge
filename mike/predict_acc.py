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


# load data 
map_dict = pk.load(open('data/map.pkl' , 'rb'))

print(map_dict)

# ========================================================
x_path = 'data/X_train.npy'
y_path = 'data/train_label.csv'
predict_path = 'predict_train_csv'
save_name = 'mike_predict_train'
# ========================================================


X = np.load(x_path)
Y_test = pd.read_csv(y_path)

# Y_test['trans'] = Y_test['label'].map(map_dict)
# Y_test['onehot'] = Y_test['trans'].apply(lambda x: to_categorical(x,num_classes=41))

Y_index = Y_test.iloc[:, 0].values
# Y_ans = Y_test['trans'].tolist()

# Y = Y_test['onehot'].tolist()


# print('X shape : ')
# print(X.shape)

# print('Y_test :')
# print(Y_test.head(3))

# print('Y onehot shape :')
# Y = np.array(Y)
# Y = Y.reshape(-1 , 41)
# print(Y.shape)



#  Normalization =====================================================
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean)/std

# predict ================================================
head = list(range(0,41))
head = [str(x) for x in head]
head.insert(0,"fname")

csv_folder = 'predict_csv'

if not os.path.exists(csv_folder):
    os.mkdir(csv_folder)

score = 0.0

for i in range(1,7):
    print('round : {}'.format(i))

    # predict
    model = load_model('model_new/mike_six_fold_{}.h5'.format(i)) 
    result = model.predict(X)
    # np.save('predict_test/mike_predict_{}.npy'.format(i),result)
    

    # write to csv 
    # result = np.load('predict_test/mike_predict_{}.npy'.format(i)) 
    df = pd.DataFrame(result)
    df.insert(0, 'ID', Y_index)
    
    df.to_csv('{}/{}_{}.csv'.format(predict_path,save_name,i), index=False,header=head)

    # calculate accuracy
    # pred = np.argmax(result, axis=-1)
    # acc = accuracy_score(Y_ans, pred)
    # print('fold {} accuracy : {}'.format(i ,acc))

    # score += result

# pred = np.argmax(score, axis=-1)
# acc = accuracy_score(Y_ans, pred)
# print('final  accuracy : {}'.format(acc))


