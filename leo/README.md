# Leo #
hello there~

## Intro of Me ##
- 孫媽lab

## Background of Me ##
- 網路資安lab: malware analysis、5G IoT technology
- know a little of DL
- new to ML

***
## Doing ##
- Integration、opening
- feature extracting
- Bagging
- first try should do the mfcc feature first
- simple baseline ensemble
- strong baseline phase1 stage2 step1: ensemble verify the unverified data
- strong baseline phase1 stage2 step2: train the xgb and lgb voting clf
- try to train ConvLSTM (?)
- Phase2 should use phase1's pre-trained model to initialize parameters to fine-tune

## Feature Extracting ##
- Using MFCC extract spectrogram of 40*345
- Using FBank extract spectrogram of 128*1034

## Bagging ##
To lower variance error
- In simple baseline phase 1 split to 30% * 3 for training and validate, 10% for testing
- In simple baseline phase 2 do the 6-fold

## Boosting ##
To lower bias error
- In phase 1 ensemble voting for stage 2
- with three kind of CNN model

## Stacking ##
- In phase 1 use the prediction of phase 1's testing output to tune stage 2
- In phase 2 using stage1's prediction of train_X to train XGB voting clf, and use stage1's pr
ediction of test_X to use XGB voting clf

## Semi-Supervised ##
- 將所有人的結果依據accuracy進行weighted-ensemble
- 僅留下confidence accuracy超過mean+std的值 或是 confidence accuracy>mean且與未進行人工verfied label相同的data
- 對原本10-fold進行fine-tuned
