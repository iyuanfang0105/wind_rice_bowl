# %%
import os
import re
import glob
import json
import jieba
import opencc
import pandas as pd

pd.set_option('display.max_columns', None)


# %%
class Dataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.traditional_to_simple_converter = opencc.OpenCC('t2s.json')

    def get_quantangshi_data_for_w2v(self, seg_type='uni'):
        tang, authors_tang_info = self.get_quantangshi_raw_data()

        if seg_type == 'jieba':
            tang['paragraphs'] = tang['paragraphs'].apply(lambda x: ' '.join(
                jieba.cut(x)))
        if seg_type == 'uni':
            tang['paragraphs'] = tang['paragraphs'].apply(lambda x: self.uni_gram(x))

        return tang, authors_tang_info

    def get_quantangshi_data_for_seq2seq(self, save_dir=''):
        tang, authors_tang_info = self.get_quantangshi_raw_data()
        text = tang['paragraphs'].values

        with open(os.path.join(save_dir, 'seq2seq_poetry_corpus.txt'), mode='w', encoding='utf-8') as f:
            for txt in text:
                input = txt.split(',')
                target = input[1:]
                target.append(input[0])
                for i, t in zip(input, target):
                    if len(i) == len(t) and (len(i) == 5 or len(i) == 7):
                        # fi.write(i + '\t' + t + '\n')
                        f.write(self.uni_gram(i) + '\t' + self.uni_gram(t) + '\n')
        # return input, target

    def get_quantangshi_raw_data(self):
        authors_tang_info = pd.read_json(os.path.join(self.data_dir, "authors.tang.json"))
        file_list_tang = glob.glob(os.path.join(self.data_dir, 'poet.tang.*.json'))
        tang = []
        for f in file_list_tang:
            tang.append(pd.read_json(f))
        tang = pd.concat(tang, axis=0)

        # replace punctuation of chinese
        tang['paragraphs'] = tang['paragraphs'].apply(lambda x: ''.join(x).replace('。', '，').replace('，', ','))

        # convert traditional to simple
        tang['paragraphs'] = tang['paragraphs'].apply(lambda x: self.traditional_to_simple_converter.convert(x.strip(',')))

        return tang, authors_tang_info

    def get_raw_quansongci_data(self):
        authors_song_info = pd.read_json(os.path.join(self.data_dir, "authors.song.json"))
        file_list_song = glob.glob(os.path.join(self.data_dir, 'poet.song.*.json'))
        song = []
        for f in file_list_song:
            song.append(pd.read_json(f))
        song = pd.concat(song, axis=0)
        return song, authors_song_info

    @staticmethod
    def save_df_to_file(df, save_path):
        with open(save_path, mode='w', encoding='utf-8') as fo:
            df.apply(lambda x: fo.write(x + '\n'))

    @staticmethod
    def uni_gram(s):
        words = []
        for w in s:
            if w != '，' and w != ',' and w != ' ':
                words.append(w)
        return ' '.join(words)


# %%
data_dir = '../data/poetry'
dataset = Dataset(data_dir)

dataset.get_quantangshi_data_for_seq2seq(save_dir=data_dir)

