# %%
import os
import csv
import itertools
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.dataset_utils import data_exploratory, feature_distribution, extract_one_hot_feature, split_dataset

pd.set_option('display.max_columns', None)
mpl.use('TkAgg')


# %%
class DSMovieLens(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.dataset = None
        self.category_features_list = ['age', 'gender', 'occupation', 'zip_code', 'release_date']
        self.multi_hot_features_map = {'film_type': ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                                                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                                                 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                                                 'Western']
                                   }
        self.features_info = {}
        self.features_primary_index = []
        self.features_secondary_index = []

    def get_dataset(self, format='ffm', save=''):
        raw_data = self.get_raw_data()
        # numerical_data = self.convert_data_to_numeric(raw_data)

        # convert date(01-Jan-1995) to year(1995)
        raw_data['release_date'] = raw_data['release_date'].apply(lambda x: self.release_date_process(x))
        raw_data = raw_data.drop(raw_data.loc[raw_data['release_date'] == '1'].index, axis=0)

        # convert age to range, (<18: "L", >=18 and <35: "M", >=35: "H", default: "N")
        raw_data['age'] = raw_data['age'].apply(lambda x: self.age_to_category(x))
        # raw_data = self.film_type_process(raw_data)

        # get list of features name and the list of features values
        self.get_features_info(raw_data)

        ds = None
        if format == 'ffm':
            ds = self.convert_to_ffm_format(raw_data)
        if format == 'lr':
            ds = self.convert_to_lr_format(raw_data)

        ds['rating'] = ds['rating'].apply(lambda x: self.rating_to_label(x))

        ds = ds.sort_values(by='timestamp')
        ds = ds.drop('timestamp', axis=1)
        train, test = split_dataset(ds, split_mode='seq', split_ratio=0.2)

        if save != '':
            self.save_df_to_file(train, os.path.join(save, 'train.txt'))
            self.save_df_to_file(test, os.path.join(save, 'test.txt'))

        return train, test, raw_data

    def get_raw_data(self):
        # data_dir = '~/WORK/wind_rice_bowl/code/wind_rice_bowl/data/movie_lens_100k'

        items = pd.read_csv(os.path.join(self.data_dir, 'u.item'),
                            names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown',
                                   'Action',
                                   'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                                   'Fantasy',
                                   'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                                   'Western'],
                            sep='|',
                            encoding='latin-1')
        # print("====>>>> items shape:\n {}".format(items.shape))
        # print("====>>>> items samples:\n {}".format(items.head()))

        users = pd.read_csv(os.path.join(self.data_dir, 'u.user'),
                            names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
                            sep='|',
                            encoding='latin-1')
        # print("====>>>> users shape:\n {}".format(users.shape))
        # print("====>>>> users samples:\n {}".format(users.head()))

        ratings = pd.read_csv(os.path.join(self.data_dir, 'u.data'),
                              names=['user_id', 'item_id', 'rating', 'timestamp'],
                              sep='\t',
                              encoding='latin-1')
        # print("====>>>> ratings shape:\n {}".format(ratings.shape))
        # print("====>>>> ratings samples:\n {}".format(ratings.head()))

        a = ratings.set_index('user_id').join(users.set_index('user_id'))
        dataset = a.set_index('item_id').join(items.set_index('movie_id'))

        dataset = dataset.drop(['video_release_date', 'IMDb_URL', 'movie_title'], axis=1)
        dataset['release_date'] = dataset['release_date'].fillna(-1)
        dataset = dataset.drop_duplicates()
        # print('====>>>> dataset shape:\n {}'.format(dataset.shape))
        # print('====>>>> dataset samples:\n {}'.format(dataset.head()))

        return dataset

    def get_features_info(self, dataset):

        for f in self.category_features_list:
            self.features_info[f] = [(f + '_' + str(i)) for i in dataset[f].unique()]

        for k in self.multi_hot_features_map.keys():
            self.features_info[k] = [k + '_' + str(i) for i in self.multi_hot_features_map[k]]

        self.features_primary_index = list(self.features_info.keys())
        self.features_secondary_index = list(itertools.chain.from_iterable(self.features_info.values()))

    def convert_to_ffm_format(self, dataset):
        for f in self.category_features_list:
            dataset[f] = dataset[f].apply(lambda x: str(self.features_primary_index.index(f)) + ":" +
                                                    str(self.features_secondary_index.index(f+"_"+str(x))) + ":" +
                                                    str(1))
        for k in self.multi_hot_features_map.keys():
            for f in self.multi_hot_features_map[k]:
                dataset[f] = dataset[f].apply(lambda x: str(self.features_primary_index.index(k)) + ":" +
                                                        str(self.features_secondary_index.index(k+"_"+f)) + ":" +
                                                        str(1) if x==1 else '')

        return dataset

    def convert_to_lr_format(self, dataset):
        return self.one_hot_coding(dataset, cols=self.category_features_list)


    def convert_data_to_numeric(self, dataset):
        """

        :param dataset: dataset df
        :return:
        """
        dataset['age'] = dataset['age'].apply(lambda x: self.age_to_category(x))
        print('====>>>> age: {}'.format(dataset['age'].unique()))

        dataset['release_date'] = dataset['release_date'].apply(lambda x: self.release_date_process(x))
        print('====>>>> release_date: {}'.format(dataset['release_date'].unique()))
        dataset = dataset.drop(dataset.loc[dataset['release_date'] == '1'].index, axis=0)
        print('====>>>> release_date: {}'.format(dataset['release_date'].unique()))

        dataset['gender'] = dataset['gender'].apply(lambda x: self.gender_maps[x])
        print('====>>>> gender: {}'.format(dataset['gender'].unique()))

        dataset['occupation'] = dataset['occupation'].apply(lambda x: self.occupation_maps[x])
        print('====>>>> occupation: {}'.format(dataset['occupation'].unique()))

        dataset['rating'] = dataset['rating'].apply(lambda x: self.rating_to_label(x))
        print('====>>>> rating: {}'.format(dataset['rating'].unique()))

        return dataset

    @staticmethod
    def one_hot_coding(dataset, cols=[]):
        for col in cols:
            dataset = pd.concat([dataset.drop([col], axis=1), pd.get_dummies(dataset[col], prefix=col)], axis=1)
        return dataset

    @staticmethod
    def release_date_process(date):
        """
        get the year from date
        :param date: 01-Jan-1995
        :return:
        """
        return str(date).strip().split('-')[-1]

    @staticmethod
    def age_to_category(age):
        """
        convert age to category
        :param age: int
        :return:
        """
        age = int(age)
        label = 'N'
        if age < 18:
            label = 'L'
        if 18 <= age < 35:
            label = 'M'
        if age >= 35:
            label = 'H'

        return label

    @staticmethod
    def rating_to_label(rating):
        rating = int(rating)
        label = -1
        if rating > 3:
            label = 1
        else:
            label = 0
        return label

    @staticmethod
    def film_type_process(dataset):
        cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                'Thriller', 'War', 'Western']
        for col in cols:
            dataset[col] = dataset[col].apply(lambda x: col if x==1 else '')

        dataset['film_type'] = dataset[cols].values.tolist()
        dataset = dataset.drop(cols, axis=1)
        dataset['film_type'] = dataset['film_type'].apply(lambda x: list(filter(None, x)))
        return dataset

    def save_df_to_file(self, df, save_path):
        with open(save_path, mode='w', encoding='utf-8') as fo:
            df.apply(lambda x: fo.write(self.row_to_str(x) + '\n'), axis=1)

    @staticmethod
    def row_to_str(row_values_list):
        res = []
        for r in row_values_list:
            if isinstance(r, list):
                for t in r:
                    if t != '':
                        res.append(str(t))
            else:
                if r != '':
                    res.append(str(r))

        return ','.join(res)

    @staticmethod
    def rating_to_label(rating):
        if rating >= 4:
            label = 1
        else:
            label = 0
        return label


# %%
if __name__ == '__main__':
    data_dir = '~/WORK/wind_rice_bowl/code/wind_rice_bowl/data/movie_lens_100k'
    # dataset = get_movielens_100k(data_dir)
    #
    # # data_exploratory(dataset)
    # train_X, train_y, test_X, test_y = extract_features(dataset)
    ds_movie_lens_100k = DSMovieLens(data_dir)

    # %%
    train, test, raw_data = ds_movie_lens_100k.get_dataset(
        format='ffm',
        save='../data/movie_lens_100k'
    )

    print()
