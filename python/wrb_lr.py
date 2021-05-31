from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from python.wrb_dataset_ad_ctr import get_ad_ctr_dataset


def train(X, y):
    classifier = LogisticRegression()
    classifier.fit(X, y)
    return classifier


def test(model, X, y):
    preds = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y.values.ravel(), preds)
    # print('====>>>> test auc: {}'.format(auc))
    return auc


if __name__ == '__main__':
    data_path = '/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl/data/ad-ctr/'
    train_X, train_y, test_X, test_y = get_ad_ctr_dataset(data_path, frac=0.01)
    print('====>>>> the shape of train_X: {}, train_y: {}, test_X: {}, test_y: {}'.format(
            train_X.shape, train_y.shape, test_X.shape, test_y.shape
        ))

    model_lr = train(train_X, train_y)
    auc = test(model_lr, test_X, test_y)
    print('====>>>> test auc: {}'.format(auc))