"""
When you observe high training accuracy, but low test accuracy, it is likely that
you encountered overfitting problem. There are in general two ways that you can
control overfitting in XGBoost:

1. The first way is to directly control model complexity. This includes：
 max_depth:

 min_child_weight:Minimum sum of instance weight (hessian) needed in a child. If the tree partition
 step results in a leaf node with the sum of instance weight less than min_child_weight, then the
 building process will give up further partitioning.

 gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree.
 The larger gamma is, the more conservative the algorithm will be.

2. The second way is to add randomness to make training robust to noise. This includes:
 subsample, colsample_bytree.

3. You can also reduce stepsize: eta. Remember to increase num_round when you do so.
"""
# %%
import os
import datetime
import calendar
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from wrb_dataset import wrb_load_iris, wrb_load_mnist


# %%
def modelfit(estimator, train_X, train_y, test_X, test_y, useTrainCV=True, cv_folds=3, early_stopping_rounds=50):
    assert len(set(test_y)) >= 2, print('label_count < 2 !!!!!!!!!!')

    metric = ['auc', 'logloss']
    if len(set(test_y)) > 2:
        metric = ['merror', 'mlogloss']

    if useTrainCV:
        xgb_param = estimator.get_xgb_params()
        xgb_param['num_class'] = 10
        xgtrain = xgb.DMatrix(train_X, label=train_y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=estimator.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        estimator.set_params(n_estimators=cvresult.shape[0])

    # #Fit the estimatororithm on the data
    # estimator.fit(train_X, dtrain['Disbursed'],eval_metric='auc')
    #
    # #Predict training set:
    # dtrain_predictions = estimator.predict(train_X)
    # dtrain_predprob = estimator.predict_proba(train_X)[:,1]
    #
    # #Print model report:
    # print "\nModel Report"
    # print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    # print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
    #
    # feat_imp = pd.Series(estimator.booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')


# %%
'''
load dataset
'''
# train_X, test_X, train_y, test_y = wrb_load_iris(visualization=False)
train_X, test_X, train_y, test_y = wrb_load_mnist(squeeze=True, sample_raio=0.04, visualization=False)

# %%
"""
Step1: Fix learning rate and n_estimators for tuning tree-based parameters,
先定下n_estimators, 一般与样本数相关
"""
params = {'objective': 'multi:softmax',
          'n_estimators': 10000,
          'learning_rate': 0.1,
          'max_depth': 5,
          'min_child_weight': 1,
          'gamma': 0,
          'subsample': 0.8,
          'colsample_bytree': 0.8,
          'nthread': 2,
          'seed': 123}

cls = xgb.XGBClassifier(**params)

modelfit(cls, train_X, train_y, test_X, test_y)

# %%
results = cls.evals_result()
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].plot(results['validation_0']['mlogloss'], label='Train')
axes[0].plot(results['validation_1']['mlogloss'], label='Test')
axes[0].legend()
axes[0].title.set_text('logloss')

axes[1].plot(results['validation_0']['merror'], label='Train')
axes[1].plot(results['validation_1']['merror'], label='Test')
axes[1].legend()
axes[1].title.set_text('merror')
plt.show()

# %%
'''
Step2: Tune max_depth and min_child_weight 
'''
