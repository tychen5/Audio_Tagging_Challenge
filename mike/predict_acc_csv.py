from keras.models import load_model
import pandas as pd 
import numpy as np 
from sklearn.model_selection import KFold
import pickle as pk 
import os

from keras.utils import to_categorical ,Sequence
import pandas as pd

from sklearn.metrics import accuracy_score


pd.options.mode.chained_assignment = None  # default='warn'
map_dict = pk.load(open('data/map.pkl' , 'rb'))
df = pd.read_csv('data/train_label.csv')



head = list(range(0,41))
head = [str(x) for x in head]
head.insert(0,"fname")

predict_path = 'predict_valid_csv'

if not os.path.exists(predict_path):
    os.mkdir(predict_path)

for k in range(1,11):
    save_name = 'mike_resnet'

    model = load_model('resnet_varified/best_{}.h5'.format(k))
    X_valid = np.load('data/ten_fold_data/X_valid_{}.npy'.format(k))
    Y_valid = np.load('data/ten_fold_data/Y_valid_{}.npy'.format(k))
    valid_fame = np.load('data/ten_fold_data/valid_fname_{}.npy'.format(k))
    result = model.predict(X_valid , verbose = 1 )

    df = pd.DataFrame(result)
    df.insert(0, 'fname', valid_fame)
    df.to_csv('{}/{}_{}.csv'.format(predict_path,save_name,k), index=False,header=head)
    
    Y_ans = np.argmax(Y_valid, axis=-1)
    pred = np.argmax(result, axis=-1)
    acc = accuracy_score(Y_ans, pred)
    print('fold {} accuracy : {}'.format(k ,acc))

