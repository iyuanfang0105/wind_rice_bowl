import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_boston, load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def wrb_load_iris(visualization=False):
    iris = load_iris()

    if visualization:
        data_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        data_df['species'] = iris.target
        data_df['species'].value_counts()
        sns.pairplot(data_df, hue="species", palette="husl", size=3, diag_kind="kde")
        plt.show()

    X = iris.data
    y = iris.target
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=7)
    print('iris dataset, train_X: {}, test_X: {}, train_y: {}, test_y: {}'.format(train_X.shape,
                                                                                  test_X.shape,
                                                                                  train_y.shape,
                                                                                  test_y.shape))
    return train_X, test_X, train_y, test_y


def wrb_load_mnist(squeeze=False, sample_raio=1, visualization=False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    train_size = x_train.shape[0]
    test_size = x_test.shape[0]

    x_train, y_train = shuffle(x_train, y_train)
    x_train = x_train[:int(np.round(sample_raio*train_size))]
    y_train = y_train[:int(np.round(sample_raio*train_size))]

    x_test, y_test = shuffle(x_test, y_test)
    x_test = x_test[:int(np.round(sample_raio*test_size))]
    y_test = y_test[:int(np.round(sample_raio*test_size))]

    if squeeze:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    print('train_x: {}, train_y: {}, test_x: {}, test_y: {}'.format(x_train.shape, y_train.shape,
                                                                    x_test.shape, y_test.shape))
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    # train_X, test_X, train_y, test_y = wrb_load_iris(visualization=True)
    wrb_load_mnist(squeeze=True, sample_raio=0.2)