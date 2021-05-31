# %%
import os
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
mpl.use('TkAgg')

# %%
# get dataset
# https://www.kaggle.com/c/avazu-ctr-prediction/data
# click-through rate prediction
def get_dataset(data_dir):
    # ata_dir = '../data/ad-ctr/'
    train_data = []
    test_data = []
    for i in range(4):
        traind = pd.read_csv(os.path.join(
            data_dir, 'train_141029' + str(i).zfill(2)))
        testd = pd.read_csv(os.path.join(
            data_dir, 'train_141030' + str(i).zfill(2)))

        train_data.append(traind)
        test_data.append(testd)

    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data)

    train_data['label'] = 1
    test_data['label'] = 2

    dataset = pd.concat([train_data, test_data])

    to_date_column(dataset)

    category_maps, dataset = category_to_numeric_according_occurrence(dataset)
    print('====>>>> dataset shape: {}'.format(dataset.shape))
    # select used cols
    cols_used = ['click', 'banner_pos', 'C1', 'weekday', 'device_type', 'site_category_idx',
                 'app_category_idx', 'C14', 'label']

    dataset_used = dataset[cols_used]
    print('====>>>> dataset_used shape: {}'.format(dataset_used.shape))
    print('====>>>> dataset_used cols: {}'.format(dataset_used.columns))
    print('====>>>> dataset_used head:\n {}'.format(dataset_used.head()))

    return dataset_used, category_maps


def category_to_numeric_according_occurrence(df, cols=('site_id', 'site_domain', 'site_category',
                                                       'app_id', 'app_domain', 'app_category', 'device_model')):
    """
    convert the category to number, according the frequency of occurrence
    :param df:
    :param cols: category names
    :return: category_index_maps and dataframe
    """
    category_maps = {}

    for col in cols:
        category_maps[col] = {}
        for id, (k, v) in enumerate(df[col].value_counts().to_dict().items()):
            category_maps[col][k] = (v, id)

        df[col +
            '_value_counts'] = df[col].apply(lambda x: category_maps[col][x][0])
        df[col + '_idx'] = df[col].apply(lambda x: category_maps[col][x][1])

    return category_maps, df


def to_date_column(df):
    df["dt_hour"] = pd.to_datetime(df["hour"], format="%y%m%d%H")
    # df["year"] = df["dt_hour"].dt.year
    # df["month"] = df["dt_hour"].dt.month
    # df["day"] = df["dt_hour"].dt.day
    df["hour"] = df["dt_hour"].dt.hour
    df["weekday"] = df["dt_hour"].dt.dayofweek
    # df["is_weekend"] = df.apply(lambda x: x["is_weekday"] in [5, 6], axis=1)
    return df


# %%
# data exploratory

def data_exploratory(dataset):
    print('====>>>> dataset describe:\n {}'.format(dataset.describe()))
    print('====>>>> dataset is null:\n {}'.format(dataset.isnull().any()))
    print('====>>>> dataset info:\n {}'.format(dataset.info()))
    print('====>>>> dataset nunique:\n {}'.format(dataset.nunique()))

    features_distribution(dataset,
                          features_name=('click', 'banner_pos', 'C1', 'weekday', 'device_type'),
                          hues_name=(None, 'click', 'click', 'click', 'click'))


def show_percentage(ax, total):
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        ax.text(p.get_x()+p.get_width()/2.,
                p.get_height() + 3,
                percentage,
                ha="center")


def features_distribution(dataset, features_name=('click', 'banner_pos'), hues_name=(None, 'click')):
    fig, axes = plt.subplots(nrows=len(features_name), ncols=2)

    train_data = dataset[dataset['label']==1]
    train_data = train_data.drop(['label'], axis=1)
    test_data = dataset[dataset['label']==2]
    test_data = test_data.drop(['label'], axis=1)

    ax_id = 0
    for f, h in zip(features_name, hues_name):
        print(f, h)
        ax = sns.countplot(x=f, hue=h, data=train_data, ax=axes[ax_id, 0])
        show_percentage(ax, train_data.shape[0])

        ax = sns.countplot(x=f, hue=h, data=test_data, ax=axes[ax_id, 1])
        show_percentage(ax, test_data.shape[0])
        ax_id = ax_id + 1
    plt.tight_layout()
    plt.show()

    # plt.subplots(121)
    # sns.heatmap(train_data.corr().abs(), annot=True)
    # plt.subplots(122)
    # sns.heatmap(test_data.corr().abs(), annot=True)
    # plt.show()


# %%
# extracting features, one-hot
def one_hot_feature(dataset):
    for col in dataset.columns:
        dataset = pd.concat([dataset.drop([col], axis=1), pd.get_dummies(
            dataset[col], prefix=col)], axis=1)
    return dataset


def extract_feats(data_df, frac=0.1):
    """

    :param data_df: dataframe, click is ground truth, label is flag for train and test
    :param frac: dataset selected ratio to build model
    :return: X, y for train and test
    """
    d = data_df[data_df.columns.difference(['click', 'label'])]
    dataset_feats = pd.concat([one_hot_feature(d), data_df[['click', 'label']]], axis=1)

    dataset_feats = dataset_feats.sample(frac=frac, replace=False, random_state=123)

    train_df = dataset_feats[dataset_feats['label'] == 1]
    train_y = train_df['click']
    train_X = train_df.drop(['click', 'label'], axis=1)

    test_df = dataset_feats[dataset_feats['label'] == 2]
    test_y = test_df['click']
    test_X = test_df.drop(['click', 'label'], axis=1)
    return train_X, train_y, test_X, test_y


# %%
# get ad-ctr-prediction dataset
def get_ad_ctr_dataset(data_path, frac=0.01):
    # data_path = '/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl/data/ad-ctr/'
    dataset, _ = get_dataset(data_path)
    print('====>>>> dataset shape: {}'.format(dataset.shape))
    print('====>>>> dataset cols: {}'.format(dataset.columns))

    train_X, train_y, test_X, test_y = extract_feats(dataset, frac=frac)
    return train_X, train_y, test_X, test_y


# %%
if __name__ == '__main__':
    data_path = '/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl/data/ad-ctr/'
    train_X, train_y, test_X, test_y = get_ad_ctr_dataset(data_path, frac=0.01)
    print('====>>>> the shape of train_X: {}, train_y: {}, test_X: {}, test_y: {}'.format(
            train_X.shape, train_y.shape, test_X.shape, test_y.shape
        ))






