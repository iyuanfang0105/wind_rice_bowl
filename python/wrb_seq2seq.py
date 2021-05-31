import os
import numpy as np
import pandas as pd
import tensorflow as tf


# %%
class DatasetTang(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.vocab = None
        self.pretrained_embedding = None

    def get_dataset(self):
        self.load_embedding()
        inputs = self.get_raw_data(os.path.join(self.data_dir, 'seq2seq_inputs.txt'))
        targets = self.get_raw_data(os.path.join(self.data_dir, 'seq2seq_target.txt'))
        print(f'====>>>> inputs: {len(inputs)}, targets: {len(targets)}')
        print(f'====>>>> examples: \n inputs: {inputs[0]}, targets: {targets[0]}')

        inputs_idx = self.txt_to_idxs(inputs)
        targets_idx = self.txt_to_idxs(targets)

        print(f'====>>>> inputs_idx: {inputs_idx.shape}, targets: {targets_idx.shape}')
        print(f'====>>>> examples: \n inputs_idx: {inputs_idx[0]}, targets_idx: {targets_idx[0]}')

        return inputs_idx, targets_idx, inputs, targets

    def txt_to_idxs(self, txt):
        """

        :param txt: a list of string, like ['公 子 申 敬 爱', '公 子 申 敬 爱']
        :return:
        """
        txt_idxs = []
        for ts in txt:
            txt_idxs.append(self.sentence_to_index(ts))
        return np.asarray(txt_idxs)

    def load_embedding(self):
        with open(os.path.join(self.data_dir, 'tang_metadata.tsv'), 'r') as mf:
            self.vocab = mf.read().split('\n')
        with open(os.path.join(self.data_dir, 'tang_vectors.tsv'), 'r') as vf:
            vetors = vf.read().split('\n')

            embd = []
            for v in vetors:
                embd.append(v.strip().split('\t'))
            self.pretrained_embedding = embd

    def sentence_to_index(self, s):
        """

        :param s: a string split by space, like: '公 子 申 敬 爱'
        :return:
        """
        s_idx = []
        for w in s.strip().split(' '):
            if w in self.vocab:
                s_idx.append(self.vocab.index(w))
            else:
                s_idx.append(self.vocab.index('[UNK]'))
        return s_idx

    @staticmethod
    def get_raw_data(data_path):
        with open(data_path, 'r') as inf:
            data = inf.read().split('\n')
        return data


# %%
data_dir = 'data/poetry'
dt = DatasetTang(data_dir)
inputs_idx, targets_idx, inputs, targets = dt.get_dataset()

# %%
dataset = tf.data.Dataset.from_tensor_slices((inputs_idx, targets_idx)).shuffle()
# dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)



