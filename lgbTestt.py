import lightgbm as lgb
import pickle
from sklearn import metrics
from sklearn.externals import joblib
import numpy as np


#traindatapickle = open('/home/songjiazhi/atpcapsule/atp388/newfeature15.pickle','rb')
#traindatapickle = open('/home/songjiazhi/atpcapsule/atp227/newfeature15.pickle','rb')
#traindatapickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/fivefold/5/label_train.pickle','rb')
#traindatapickle = open('/home/songjiazhi/atpcapsule/atp227/seperatefeature/fivefold/5/label_train.pickle','rb')
traindatapickle = open('/home/songjiazhi/atpcapsule/paper/traindata/label.pickle','rb')
traindata = pickle.load(traindatapickle)
#feature_train = traindata[1]
#label_train = traindata[0]
label_train = traindata

#feature_train_pickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/commonfeature.pickle','rb')
#feature_train_pickle = open('/home/songjiazhi/atpcapsule/atp227/seperatefeature/commonfeature.pickle','rb')
#feature_train_pickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/fivefold/5/commonfeature_train.pickle','rb')
#feature_train_pickle = open('/home/songjiazhi/atpcapsule/atp227/seperatefeature/fivefold/5/commonfeature_train.pickle','rb')
#feature_train_pickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/fivefold/5/pssmonehotfeature_train.pickle','rb')
feature_train_pickle = open('/home/songjiazhi/atpcapsule/paper/traindata/commonfeature.pickle','rb')
feature_train = pickle.load(feature_train_pickle)


testdatapickle = open('/home/songjiazhi/atpcapsule/atp41/newfeature15.pickle','rb')
#testdatapickle = open('/home/songjiazhi/atpcapsule/atp17/newfeature15.pickle','rb')
#testdatapickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/fivefold/5/label_test.pickle','rb')
#testdatapickle = open('/home/songjiazhi/atpcapsule/atp227/seperatefeature/fivefold/5/label_test.pickle','rb')
testdata = pickle.load(testdatapickle)
#feature_test = testdata[1]
label_test = testdata[0]
#label_test = testdata

feature_test_pickle = open('/home/songjiazhi/atpcapsule/atp41/seperatefeature/commonfeature.pickle','rb')
#feature_test_pickle = open('/home/songjiazhi/atpcapsule/atp17/seperatefeature/commonfeature.pickle','rb')
#feature_test_pickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/fivefold/5/commonfeature_test.pickle','rb')
#feature_test_pickle = open('/home/songjiazhi/atpcapsule/atp227/seperatefeature/fivefold/5/commonfeature_test.pickle','rb')
#feature_test_pickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/fivefold/5/pssmonehotfeature_test.pickle','rb')
feature_test = pickle.load(feature_test_pickle)

#print(feature_train[17:19])

lgb_train = lgb.Dataset(feature_train, label_train)
lgb_eval = lgb.Dataset(feature_test, label_test, reference=lgb_train)

params = {
'boosting_type':'gbdt',
'objective':'binary',
'metric':{'auc'},
#'metric':{'auc','binary_logloss'}
'num_leaves':128,
'learning_rate':0.05,
'feature_fraction':0.8,
'bagging_fraction':0.8,
'bagging_freq':5,
'verbose':0,
'num_threads':8,
'is_unbalance':True
}
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=500,
                valid_sets=[lgb_train, lgb_eval],
                valid_names=['train','valid'],
                #early_stopping_rounds=100,
                verbose_eval=1)
joblib.dump(gbm, '/home/songjiazhi/atpcapsule/paper/lgb_model.m')
#modelpickle = open('/home/songjiazhi/atpcapsule/atp227/lgb_model.m','wb')
#modelpickle = open('/home/songjiazhi/atpcapsule/atp388/fivefold/5/lgb_model.m','wb')
#modelpickle = open('/home/songjiazhi/atpcapsule/atp227/fivefold/5/lgb_model.m','wb')
#modelpickle = open('/home/songjiazhi/atpcapsule/atp388/seperatefeature/onehot_lgb.m','wb')
#pickle.dump(gbm,modelpickle)
#prediction = gbm.predict(feature_test, num_iteration=gbm.best_iteration)
#print(prediction)
#predictionpickle = open('D:\\atpcapsule\\lightgbmprediction.pickle','wb')
#pickle.dump(prediction,predictionpickle)

#print(prediction)
#predictionpickle = open('/home/songjiazhi/atpbinding/atp227/lgbprediction/1/prediction.pickle','rb')
#pickle.dump(prediction, predictionpickle)
#length = len(label_test)
##for i in range(length):
    ##print(prediction[i], label_test[i])
#print(metrics.roc_auc_score(label_test,prediction))


