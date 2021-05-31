import os
import datetime
import calendar
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
import xgboost as xgb
import matplotlib.pyplot as plt
import xlearn as xl

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.datasets import dump_svmlight_file


def get_movie_lens_25m(data_dir):
    file_list = os.listdir(data_dir)
    data_set = {}

    for f in file_list:
        if f.endswith('.csv'):
            f_name = f.split('.')[0]
            if f_name not in data_set.keys():
                data_set[f_name] = pd.read_csv(os.path.join(data_dir, f))
    return data_set


def get_movie_lens_100k(data_dir, debug=False):
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

    print(b.columns)
    dataset = b[['rating', 'age', 'gender', 'occupation',
                 'release_date', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']]

    # make label for the task for CTR prediction
    dataset.loc[dataset['rating'] <= 3, 'rating'] = 0
    dataset.loc[dataset['rating'] > 3, 'rating'] = 1

    if debug:
        dataset = dataset.sample(1000)

    # dataset = dataset.dropna(axis=0, how='any')
    # dataset = encode_one_hot(dataset, cols_name=['gender', 'occupation'])

    print('====>>>> dataset shape: {}'.format(dataset.shape))
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
    print('====>>>> dataset shape: {}'.format(dataset.shape))

    y_df = dataset['rating']
    X_df = dataset.drop('rating', axis=1)
    # dump_svmlight_file(X_df, y_df, os.path.join(data_dir, 'dataset.libsvm'))

    y = np.asarray(y_df.values, dtype=np.float)
    X = np.asarray(X_df.values, dtype=np.float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=42)
    print('====>>>> X_train: {}, y_train: {}, X_test: {}, y_test: {}'.format(X_train.shape, y_train.shape,
                                                                             X_test.shape, y_test.shape))
    return X_train, X_test, y_train, y_test


def encode_one_hot(dataframe, cols_name=[]):
    for col in cols_name:
        one_hot = pd.get_dummies(dataframe[col], prefix=col)
        dataframe = dataframe.drop(col, axis=1)
        dataframe = pd.concat([dataframe, one_hot], axis=1)

    return dataframe


def gbdt_lr(X_train, y_train, X_test, y_test):
    n_estimators = 50

    gbdt = GradientBoostingClassifier(n_estimators=n_estimators, random_state=10, subsample=0.6, max_depth=7,
                                      min_samples_split=10)
    gbdt.fit(X_train, y_train)
    y_pred = gbdt.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print('auc of model gbdt: {}'.format(auc))

    # extract features
    train_new_feature = gbdt.apply(X_train).reshape(-1, n_estimators)
    test_new_feature = gbdt.apply(X_test).reshape(-1, n_estimators)

    enc = OneHotEncoder()
    enc.fit(np.concatenate([train_new_feature, test_new_feature]))

    # # 每一个属性的最大取值数目
    # print('每一个特征的最大取值数目:', enc.n_values_)
    # print('所有特征的取值数目总和:', enc.n_values_.sum())

    train_new_feature_1 = np.array(enc.transform(train_new_feature).toarray())
    test_new_feature_1 = np.array(enc.transform(test_new_feature).toarray())
    print(
        'features transformed by gbdt, trian: {}, test: {}'.format(train_new_feature_1.shape, test_new_feature_1.shape))

    # lr model
    lr = LogisticRegression()
    lr.fit(train_new_feature_1, y_train)

    y_pred_lr = lr.predict_proba(test_new_feature_1)[:, 1]
    auc_lr = roc_auc_score(y_test, y_pred_lr)
    print('auc of model gbdt+lr: {}'.format(auc_lr))

    return 0


def xgboost(X_train, y_train, X_test, y_test, num_boost_round=999):
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_test = xgb.DMatrix(X_test, label=y_test)

    params = {
        'max_depth': 6,  # default=6
        'min_child_weight': 1,  # default=1
        'gamma': 0,  # default=0
        'subsample': 1,  # sampling rate for data, default=1
        'colsample_bytree': 1,  # sampling rate for cols (features), default=1
        'eta': 0.3,  # lr, default=0.3
        'lambda': 1,  # L2, default=1
        'alpha': 0,  # L1, default=0
        'objective': 'binary:logistic',
        'eval_metric': 'auc'
    }

    model = xgb.train(params, d_train, num_boost_round=num_boost_round, evals=[(d_test, "Test")], early_stopping_rounds=15)

    gridsearch_params = [
        (max_depth, min_child_weight)
        for max_depth in range(3, 10)
        for min_child_weight in range(1, 12)
    ]

    min_mae = float("Inf")
    best_params = None
    for max_depth, min_child_weight in gridsearch_params:
        print("CV with max_depth={}, min_child_weight={}".format(max_depth, min_child_weight))
        # Update our parameters
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        # Run CV
        cv_results = xgb.cv(
            params,
            d_train,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            early_stopping_rounds=15
        )
        # Update best MAE
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (max_depth, min_child_weight)
    print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))






if __name__ == '__main__':
    # get dataset and features
    data_dir = '../data/movie_lens_100k'
    X_train, X_test, y_train, y_test = get_movie_lens_100k(data_dir, debug=True)

    # build model
    # gbdt_lr(X_train, y_train, X_test, y_test)
    xgboost(X_train, y_train, X_test, y_test)
