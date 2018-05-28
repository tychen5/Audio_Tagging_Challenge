import pandas as pd 
import numpy as np 
from sklearn.model_selection import KFold
import pickle as pk 
import os

from keras.utils import to_categorical ,Sequence

audio_martix = np.load('X_train_verified_3710.npy')
