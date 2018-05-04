'''
reference :
https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data
'''

import numpy as np 

# five fold for validation 
X_train = np.load('data/mfcc/X_train.npy')
X_test = np.load('data/mfcc/X_test.npy')

print('X_train shape : ')
print(X_train.shape)

print('X_test shape : ')
print(X_test.shape)

'''
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std
'''



