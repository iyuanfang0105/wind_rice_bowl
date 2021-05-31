import os
import pandas as pd
import matplotlib.pyplot as plt
import xlearn as xl

from sklearn.metrics import roc_auc_score


class WrbFM(object):
    def __init__(self):
        print('init fm')

    @staticmethod
    def train(X, y):
        model = xl.FMModel(task='binary', init=0.1,
                           epoch=10, k=4, lr=0.1,
                           reg_lambda=0.01, opt='sgd',
                           metric='auc')
        model.fit(X, y)
        # print model weights
        print('====>>>> weights of FM-Model: {}'.format(model.weights))
        return model

    @staticmethod
    def test(model, X, y):
        preds = model.predict(X)
        auc = roc_auc_score(y.values.ravel(), preds)
        return auc


class WrbFFM(object):
    def __init__(self, train_data_path, test_data_path, save_path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.save_path = save_path

    def train(self):
        # Training task
        ffm_model = xl.create_ffm()                # Use field-aware factorization machine (ffm)
        ffm_model.setTrain("../data/movie_lens_100k/train.txt")    # Set the path of training dataset
        ffm_model.setValidate("../data/movie_lens_100k/test.txt")  # Set the path of validation dataset

        # Parameters:
        #  0. task: binary classification
        #  1. learning rate: 0.2
        #  2. regular lambda: 0.002
        #  3. evaluation metric: accuracy
        param = {'task': 'binary', 'lr': 0.2, 'lambda': 0.002, 'metric': 'auc'}

        # Start to train
        # The trained model will be stored in model.out
        if self.save_path != '':
            ffm_model.fit(param, os.path.join(self.save_path, 'ffm.model'))
        return ffm_model

    def test(self):
        ffm_model = xl.create_ffm()
        # Prediction task
        ffm_model.setTest(self.test_data_path)  # Set the path of test dataset
        ffm_model.setSigmoid()                 # Convert output to 0-1

        # Start to predict
        # The output result will be stored in output.txt
        res = ffm_model.predict(os.path.join(self.save_path, 'ffm.model'))
        print()


def get_movie_len_100k(data_dir, show=False):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    u_data = pd.read_csv(os.path.join(dataset_dir, 'u.data'), sep='\t', names=names)

    num_users = u_data.user_id.unique().shape[0]
    num_items = u_data.item_id.unique().shape[0]

    if show:
        plt.hist(u_data['rating'], bins=5, ec='black')
        plt.xlabel('rating')
        plt.ylabel('count')
        plt.title('Distribution of Ratings in MovieLens 100K')
        plt.show()
    return u_data




if __name__ == '__main__':
    dataset_dir = '../data/movie_lens_100k/'
    u_data = get_movie_len_100k(dataset_dir, show=True)


