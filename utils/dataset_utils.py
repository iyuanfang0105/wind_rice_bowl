import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
mpl.use('TkAgg')


def data_exploratory(dataset):
    """
    show basic infomation of dataset
    :param dataset: dataset dataframe
    :return:
    """
    print('====>>>> dataset describe:\n {}'.format(dataset.describe()))
    print('====>>>> dataset is null:\n {}'.format(dataset.isnull().any()))
    # print('====>>>> dataset info:\n {}'.format(dataset.info()))
    print('====>>>> dataset nunique:\n {}'.format(dataset.nunique()))


def show_percentage(ax, total):
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        ax.text(p.get_x()+p.get_width()/2.,
                p.get_height() + 3,
                percentage,
                ha="center")


def feature_distribution(dataset, feature_name='', hue_name=''):
    """
    show the distribution of features col
    :param dataset: dataframe
    :param feature_name: col to be shown
    :param hue_name: col for hue
    :return:
    """

    if feature_name != '':
        if hue_name != '':
            ax = sns.countplot(x=feature_name, hue=hue_name, data=dataset)
        else:
            ax = sns.countplot(x=feature_name, hue=None, data=dataset)
        show_percentage(ax, dataset.shape[0])


def extract_one_hot_feature(dataset, cols=[]):
    """
    convert category feature to one hot
    :param dataset: dataframe
    :param cols: list of cols
    :return:
    """
    for col in cols:
        dataset = pd.concat([dataset.drop([col], axis=1), pd.get_dummies(
            dataset[col], prefix=col)], axis=1)
    return dataset


def split_dataset(dataset, split_mode='random', split_ratio=0.2):
    """
    split dataset into train and test according random and seq-aware mode
    :param dataset: dataset df
    :param split_mode: random and seq-aware, if using seq-aware mode, the date should be sorted
    :param split_ratio: ratio of train and test
    :return:
    """
    train = []
    test = []
    if split_mode == 'random':
        train, test = train_test_split(dataset, test_size=split_ratio, shuffle=True, random_state=42)
    if split_mode == 'seq':
        split_row_id = round(dataset.shape[0] * split_ratio)
        train = dataset.iloc[:split_row_id, :]
        test = dataset.iloc[split_row_id:, :]

    return train, test

def convert_csv_to_libffm():
    return 0
