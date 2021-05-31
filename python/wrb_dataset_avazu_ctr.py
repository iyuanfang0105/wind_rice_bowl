import os
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt

dataset_dir = '/Users/wind/WORK/public_data_set/avazu-ctr-prediction/'
train = pd.read_csv(os.path.join(dataset_dir, 'train.gz'), compression='gzip')
print(train.head())
print('train shape: {}'.format(train.shape))

# split the train data according date
for d in range(21, 31):
    st = int('1410' + str(d) + '00')
    ed = int('1410' + str(d) + '23')
    data = train[(train['hour'] >= st) & (train['hour'] <= ed)]
    print('data from {} to {}: {}'.format(st, ed, data.shape))
    data.to_csv(os.path.join('../data/avazu_ctr/train', 'data_' + str(d) + '.csv'), index=False, compression='gzip')



