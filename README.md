# ML2018SPRING_finalProject
大家都去當大神惹QQ_NTU-EE-ML2018SPRING_Kaggle競賽期末專題
***
* 太大的檔案放雲端: https://drive.google.com/drive/folders/1LdcR5EPBbYdcC8ltlp83B-deS78xOYOL?usp=sharing
* 僅保留master branch，各自所做的事情放自己的資料夾~
* 整理好的code與report放final繳交資料夾

## Announcement Phase1 ##
- 大家辛苦了~嘎U，記得保留model跟上傳code
- phase1 stage1: 各自對train_X及verified label進行shuffle切10 fold(不train unverified的)並記錄好各個model所用來training跟valid/predict的data分別是誰
- 將訓練好的model之該valid fold data ACC填至表單 ( https://docs.google.com/spreadsheets/d/1vTM0qrHe3V_AaR0PKjnblsMA6EKb73mSNiEIjgwyy70/edit#gid=0 ) ，希望此次大家各自的10個model的ACC至少可以有0.65~0.7，可能要多嘗試一些參數來tune
- jerry: RAW + LSTM/GRU model
- mow: MFCC + CNN2D model
- mike: RAW + CNN1D model
- 每個人用自己的10個model predict該fold的validation data以及所有unverified data，所以每個人共需要上傳20個csv
- 使用MFCC/RAW於google drive feature_all/中的train_X來訓練model
- 全部的label放在feature_all資料夾中
- 6/2 中午以前上傳好10-fold CV predict結果的csv(20個檔案)到google drive: feature_all/predict_valid_data & feature_all/predict_unverified_data
  
  - csv header格式為fname,。(csv的第一個row麻煩寫入header，命名為自己英文名字開頭)
  
  - prob為softmax predict出來結果的1D numpy array，shape為(41,)。依照map.pkl的數字順序，第一個數字代表該fname是屬於map.pkl第0類的機率(也就是Hi-hat)，第二個數字代表該fname是屬於1類的機率(Saxophone)，第三個數字代表該fname是2的機率...etc
    
  - 存成csv的時候別把row的index值存進去，只要fname,prob就好了，THX
  
  - training時請注意各fold的label順序是否一致，predict valid data時記得寫入該對應的fname
  

- 5/31 晚上1900 教研館319討論遇到的問題、IDEA、創意、架構、方法改良~~

## 版本請注意 ##
*基本上皆採用最新的，這樣最單純以免合不起來*
- scikit-learn 0.19.1
- scipy 1.0.1
- librosa 0.6.0

- tensorflow-gpu 1.8
- keras 2.1.6
- xgboost 0.71
- lightgbm 2.1
- catboost 0.8.1

## Voice ##
* 最少樣本的類別有94個 (一種)
* 最多樣本的類別有300個 (三分之一種類)
* 每秒有44100個點
* 一個點有65536種可能
* audio length在不同類別會不一樣(可當成一種feature)
* 需要把outlier長度的audio踢掉(1.5*iqr / 95%)
* data normalized到0~1之間 (-min)/(max-min)
* 10-fold CV (sklearn.cross_validation.StratifiedKFold)=>ensemble/bagging
* MFCC/FBANK
* data augmentation
* CNN1D/CNN2D/LSTM/GRU/ConvLSTM
* random foreset clf/ XGB clf(gpu)/ catboost clf (gpu) => https://www.kaggle.com/mainya/stacking-xgboost-lightgbm-catboost 
* train on manually labeled or two times, and train on other label
* weakly-supervised learning / unsupervised clustering (kick outlier)
* PCA/MF/SVD =>轉折點
* self-training / stacking
* sklearn.preprocessing import StandardScaler  / MinMaxScaler / normalize (including train_X & test_X)
* def extract_features(files, path): in kernel notebook


## 環境安裝 ## 
```bash
    $ pip install -r requirements.txt
```
