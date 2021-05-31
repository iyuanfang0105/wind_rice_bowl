# %%
import python.wrb_dataset_ad_ctr as db_ad_ctr
import python.wrb_lr as algo_lr
import python.wrb_xgboost as algo_xgb

from python.wrb_dataset_movie_lens_100k import DSMovieLens
from python.wrb_fm_ffm import WrbFM, WrbFFM


# %%
def get_dateset(name=''):
    train_X, train_y, test_X, test_y = [], [], [], []

    if name == 'ad_ctr':
        data_path = '/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl/data/ad-ctr/'
        train_X, train_y, test_X, test_y = db_ad_ctr.get_ad_ctr_dataset(data_path, frac=0.01)
        print('====>>>> the shape of train_X: {}, train_y: {}, test_X: {}, test_y: {}'.format(
                train_X.shape, train_y.shape, test_X.shape, test_y.shape
            ))
    if name == 'movie_lens':
        data_dir = '~/WORK/wind_rice_bowl/code/wind_rice_bowl/data/movie_lens_100k'
        ds_movie_lens_100k = DSMovieLens(data_dir)
        ds_movie_lens_100k.get_dataset(format='ffm', save='../data/movie_lens_100k')
        train, test, raw_data = ds_movie_lens_100k.get_dataset(format='lr')

        train_y = train['rating']
        train_X = train.drop('rating', axis=1)

        test_y = test['rating']
        test_X = test.drop('rating', axis=1)

    return train_X, train_y, test_X, test_y


# %%
def build_model(train_X, train_y, test_X, test_y, algo_name='lr'):
    model = None
    if algo_name == 'lr':
        model = algo_lr.train(train_X, train_y)
        auc = algo_lr.test(model, test_X, test_y)
        print('====>>>> AUC-LR: {}'.format(auc))

    if algo_name == 'xgboost':
        model = algo_xgb.train(train_X, train_y)
        auc = algo_xgb.test(model, test_X, test_y)
        print('====>>>> AUC-Xgboost: {}'.format(auc))

    if algo_name == 'fm':
        algo_fm = WrbFM()
        model = algo_fm.train(train_X, train_y)
        auc = algo_fm.test(model, test_X, test_y)
        print('====>>>> AUC-FM: {}'.format(auc))

    if algo_name == 'ffm':
        algo_ffm = WrbFFM(train_data_path='../data/movie_lens_100k/train.txt',
                          test_data_path='../data/movie_lens_100k/test.txt',
                          save_path='./')
        algo_ffm.train()
        # algo_ffm.test()


# %%
if __name__ == '__main__':
    train_X, train_y, test_X, test_y = get_dateset(name='movie_lens')
    build_model(train_X, train_y, test_X, test_y, algo_name='lr')
    build_model(train_X, train_y, test_X, test_y, algo_name='xgboost')

    build_model(train_X, train_y, test_X, test_y, algo_name='fm')
    build_model(train_X, train_y, test_X, test_y, algo_name='ffm')