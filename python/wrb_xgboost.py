"""
When you observe high training accuracy, but low test accuracy, it is likely that
you encountered overfitting problem. There are in general two ways that you can
control overfitting in XGBoost:

1. The first way is to directly control model complexity. This includes：
 max_depth [default=6]:

 min_child_weight  [default=1], [0,∞]: Minimum sum of instance weight (hessian) needed in a child. If the tree partition
 step results in a leaf node with the sum of instance weight less than min_child_weight, then the
 building process will give up further partitioning.

 gamma [default=0]  [0,∞]: Minimum loss reduction required to make a further partition on a leaf node of the tree.
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
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score

from wrb_dataset import wrb_load_iris, wrb_load_mnist


def get_movie_lens_100k(data_dir, sample_ratio=1.0):
    # file_names = ['u.data', 'u.genre', 'u.item']
    items = pd.read_csv(os.path.join(data_dir, 'u.item'),
                        names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown',
                               'Action',
                               'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                               'Fantasy',
                               'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                               'Western'],
                        sep='|',
                        encoding='latin-1')

    users = pd.read_csv(os.path.join(data_dir, 'u.user'),
                        names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
                        sep='|',
                        encoding='latin-1')

    ratings = pd.read_csv(os.path.join(data_dir, 'u.data'),
                          names=['user_id', 'item_id', 'rating', 'timestamp'],
                          sep='\t',
                          encoding='latin-1')

    a = ratings.set_index('user_id').join(users.set_index('user_id'))
    b = a.set_index('item_id').join(items.set_index('movie_id'))

    # print(b.columns)
    dataset = b[['rating', 'age', 'gender', 'occupation',
                 'release_date', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']]

    # make label for the task for CTR prediction
    dataset.loc[dataset['rating'] <= 3, 'rating'] = 0
    dataset.loc[dataset['rating'] > 3, 'rating'] = 1

    print('====>>>> original dataset shape: {}'.format(dataset.shape))
    dataset = dataset.sample(int(np.floor(dataset.shape[0] * sample_ratio)))

    # dataset = dataset.dropna(axis=0, how='any')
    # dataset = encode_one_hot(dataset, cols_name=['gender', 'occupation'])

    print('====>>>> sampled dataset shape: {}'.format(dataset.shape))
    # print('====>>>> dataset sample: \n{}'.format(dataset.sample(2)))

    # quantify 'occupation' and 'gender'
    occupation_map = dataset['occupation'].unique()
    gender_map = dataset['gender'].unique()
    dataset['occupation'] = dataset['occupation'].apply(lambda x: np.where(occupation_map == x)[0][0])
    dataset['gender'] = dataset['gender'].apply(lambda x: np.where(gender_map == x)[0][0])

    # convert date to the day of week
    dataset.release_date = dataset.release_date.str.split('-')
    dataset = dataset.dropna(axis=0, how='any')
    month_abbr_to_num = {month: index for index, month in enumerate(calendar.month_abbr) if month}
    dataset['week'] = dataset['release_date'].apply(
        lambda x: datetime.date(int(x[2]), month_abbr_to_num[x[1]], int(x[0])).weekday())
    dataset = dataset.drop('release_date', axis=1)

    # y_df = dataset['rating']
    # X_df = dataset.drop('rating', axis=1)
    # dump_svmlight_file(X_df, y_df, os.path.join(data_dir, 'dataset.libsvm'))

    # y = np.asarray(y_df.values, dtype=np.float)
    # X = np.asarray(X_df.values, dtype=np.float)
    train, test = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=42)
    # print('====>>>> X_train: {}, y_train: {}, X_test: {}, y_test: {}'.format(X_train.shape, y_train.shape,
    #                                                                          X_test.shape, y_test.shape))
    print('****>>>> train: {}, test: {}'.format(train.shape, test.shape))
    return train, test


def grid_search(params, gs_params, train_X, train_y, eval_X, eval_y, eval_metric='roc-auc'):
    """
    grid search for xgboost
    :param params: dict, original params of xgboost
    :param gs_params: dict, need to be tunning params
    :param X: input X, numpy array
    :param y: input y, numpy array
    :param eval_metric: f1_macro for multi-classes, roc_auc for binary-class
    :return: updated params
    """

    estimator = xgb.XGBClassifier(**params)
    # estimator.fit(train_X, train_y, eval_set=[(eval_X, eval_y)], eval_metric=scoring,
    #               early_stopping_rounds=10, verbose=True)
    # estimator.set_params(n_estimators=estimator.best_ntree_limit)

    gsearch = GridSearchCV(estimator=estimator, param_grid=gs_params, scoring=eval_metric, cv=3, verbose=True, n_jobs=2)
    gsearch.fit(train_X, train_y)

    best_params = gsearch.best_params_
    best_score = gsearch.best_score_
    print('****>>>> best params: {}, best scores: {}'.format(best_params, best_score))

    for p in best_params.keys():
        if p in params.keys():
            params[p] = best_params[p]
        else:
            print('****>>>> Error: the param {} do not exist'.format(p))

    return params


def train(train_X, train_y, eval_X, eval_y, task_type='binary_class'):
    label_count = len(set(train_y))
    assert label_count >= 2, print('label_count should be >=2 !!!!!')

    # for xgb param
    objective = 'binary:logistic'
    xgb_metric = 'logloss'
    if label_count > 2:
        objective = 'multi:softmax'
        xgb_metric = 'mlogloss'

    # for sickit-learn param
    sk_metric = 'roc-auc'
    if label_count > 2:
        sk_metric = 'f1_macro'

    # initial params
    sk_params = {
        'objective': objective,  # binary:logistic, multi:softmax
        'n_estimators': 10000,
        'learning_rate': 0.1,  # lr, default=0.3 , called 'eta' in xgboost

        'max_depth': 6,  # default=6
        'min_child_weight': 1,  # default=1
        'gamma': 0,  # default=0

        'subsample': 1,  # sampling rate for data, default=1
        'colsample_bytree': 1,  # sampling rate for cols (features), default=1

        'reg_lambda': 1,  # L2, default=1, called 'lamda' in xgboost
        'reg_alpha': 0,  # L1, default=0, called 'alpha' in xgboost
    }


    # grid-search params
    grid_search_params = [
        {'max_depth': [4,6,8], 'min_child_weight': [0.5, 1, 1.5], 'gamma': [0, 0.5, 1]},
        # {'subsample':[i/10.0 for i in range(1,10)], 'colsample_bytree':[i/10.0 for i in range(1,10)]},
        # {'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05], 'reg_lambda':[0, 0.001, 0.005, 0.01, 0.05]}
    ]

    for id, gs_param in enumerate(grid_search_params):
        print('====>>>> Step: {}, tuning params: {}'.format(id, gs_param.keys()))
        print('++++>>>> fixing n_estimators')
        estimator = xgb.XGBClassifier(**sk_params)
        estimator.fit(train_X, train_y, eval_set=[(eval_X, eval_y)], eval_metric=xgb_metric,
                      early_stopping_rounds=10, verbose=True)
        # estimator.set_params(n_estimators=estimator.best_ntree_limit)
        sk_params['n_estimators'] = estimator.best_ntree_limit

        print('++++>>>> grid searching')
        sk_params = grid_search(sk_params, gs_param, train_X, train_y, eval_X, eval_y, eval_metric=sk_metric)
    print('====>>>> selected params: {}'.format(sk_params))

    # train the model using selected parameters
    clf = xgb.XGBClassifier(**sk_params)
    clf.fit(train_X, train_y, eval_set=[(eval_X, eval_y)], eval_metric=xgb_metric,
            early_stopping_rounds=10, verbose=True)

    features_importance = clf.feature_importances_
    features_names = clf.get_booster().feature_names
    feats_imp = pd.DataFrame({'feature_names': features_names, 'importance': features_importance}).sort_values(
        by='importance', ascending=False).set_index('feature_names')
    feats_imp.iloc[:30, :].plot.bar()
    plt.ylabel('Feature Importance Score')
    plt.show()

    return clf


def test(model, X, y):
    preds = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y.values.ravel(), preds)
    # print('====>>>> test auc: {}'.format(auc))
    return auc


# def model_fit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
#     if useTrainCV:
#         xgb_param = alg.get_xgb_params()
#         xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
#         cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
#                           metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
#         alg.set_params(n_estimators=cvresult.shape[0])
#
#     # Fit the algorithm on the data
#     alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')
#
#     # Predict training set:
#     dtrain_predictions = alg.predict(dtrain[predictors])
#     dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # # Print model report:
    # print
    # "\nModel Report"
    # print
    # "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    # print
    # "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
    #
    # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')


# def model_fit(estimator, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
#     if useTrainCV:
#         xgb_param = estimator.get_xgb_params()
#         xgtrain = xgb.DMatrix(X, label=y)
#         cvresult = xgb.cv(xgb_param, xgtrain,
#                           num_boost_round=estimator.get_params()['n_estimators'],
#                           nfold=cv_folds,
#                           metrics='auc',
#                           early_stopping_rounds=early_stopping_rounds,
#                           show_progress=False)
#         estimator.set_params(n_estimators=cvresult.shape[0])
#
#     # Fit the estimatororithm on the data
#     estimator.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')
#
#     # Predict training set:
#     dtrain_predictions = estimator.predict(dtrain[predictors])
#     dtrain_predprob = estimator.predict_proba(dtrain[predictors])[:, 1]
#
#     # # Print model report:
#     # print
#     # "\nModel Report"
#     # print
#     # "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
#     # print
#     # "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
#     #
#     # feat_imp = pd.Series(estimator.booster().get_fscore()).sort_values(ascending=False)
#     # feat_imp.plot(kind='bar', title='Feature Importances')
#     # plt.ylabel('Feature Importance Score')



# %%
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # train_X, test_X, train_y, test_y = wrb_load_iris(visualization=False)
    train_X, test_X, train_y, test_y = wrb_load_mnist(squeeze=True, sample_raio=0.02, visualization=False)
    clf = train(train_X, train_y, test_X, test_y)

    # params = {'n_estimators': 10000, 'learning_rate': 0.1}
    # cls = xgb.XGBClassifier(**params)
    # eval_set = [(train_X, train_y), (test_X, test_y)]
    #
    # cls.fit(train_X, train_y, eval_metric=['merror', 'mlogloss'], early_stopping_rounds=10, eval_set=eval_set, verbose=True)
    # cls.set_params(n_estimators = cls.best_ntree_limit)
# %%
#     results = cls.evals_result()
#     # epochs = len(results['validation_0']['mlogloss'])
#     # x_axis = range(0, epochs)
#     # plot log loss
#     # fig, ax = plt.subplots()
#     # ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
#     # ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
#     # ax.legend()
#     # plt.ylabel('Log Loss')
#     # plt.title('XGBoost Log Loss')
#     # plt.show()
#     fig, axes = plt.subplots(nrows=1, ncols=2)
#     axes[0].plot(results['validation_0']['mlogloss'], label='Train')
#     axes[0].plot(results['validation_1']['mlogloss'], label='Test')
#     axes[0].legend()
#     axes[0].title.set_text('logloss')
#
#     axes[1].plot(results['validation_0']['merror'], label='Train')
#     axes[1].plot(results['validation_1']['merror'], label='Test')
#     axes[1].legend()
#     axes[1].title.set_text('merror')
#     plt.show()