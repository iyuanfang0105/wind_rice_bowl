import re
import jieba

import pandas as pd
from tqdm import tqdm

from collections import Counter

punctuation = "，。！……？（）【】~!@#$%^&*()_+`{}|\[\]\:\";\-\\\='<>?,./"



def convert_waimai_data_to_uniform_txt_corpus(file_path, is_remove_punctuation=True):
    df = pd.read_csv(file_path)
    X = df['review']
    y = df['label']

    # remove the punctuation
    X_refined = []
    for s in X:
        if is_remove_punctuation:
            s = re.sub(r'[{}]+'.format(punctuation), ' ', s)
        s = [w for w in jieba.cut(s) if w.strip()]
        X_refined.append(s)
    return X_refined, y


class Corpus(object):
    def __init__(self, corpus, vocab_size=5000):
        self.vocab_size = vocab_size
        self.corpus = corpus
        self.word_to_id, self.id_to_word, self.word_frequency = self.build_vocab()

    def build_vocab(self):
        '''
        build vocabulary for corpus
        :return:
        word_frequncy
        word_to_id
        id_to_word
        '''
        words = []
        word_to_id = {'<UNK>': 0, '<START>': 1, '<END>': 2, '<PAD>': 3}
        id_to_word = {0: '<UNK>', 1: '<START>', 2: '<END>', 3: '<PAD>'}
        for sentence in self.corpus:
            for w in sentence:
                words.append(w)
        word_frequncy = Counter(words).most_common(self.vocab_size - len(word_to_id))
        for idx, (k, v) in enumerate(word_frequncy):
            word_to_id[k] = idx + 4
            id_to_word[idx+4] = k
        return word_to_id, id_to_word, word_frequncy

    def encoding_corpus_to_index(self, max_sentence_len):
        '''

        :param max_sentence_len:
        :return:
        '''
        corpus_to_id = []
        for s in self.corpus:
            s_refined = []
            for w in s[:max_sentence_len]:
                if w in self.word_to_id.keys():
                    s_refined.append(self.word_to_id[w])
                else:
                    s_refined.append(self.word_to_id['<UNK>'])
            if len(s_refined) < max_sentence_len:
                for i in range(max_sentence_len-len(s_refined)):
                    s_refined.append(self.word_to_id['<PAD>'])
            corpus_to_id.append(s_refined)
        return corpus_to_id


def get_waimai_dataset(file_path, is_remove_punctuation=True, vocab_size=1000, max_sentence_len=15, padding=True):
    df = pd.read_csv(file_path)
    review = df['review']

    # remove the punctuation, build vocab
    review_split = []
    words = []
    for s in tqdm(review):
        if is_remove_punctuation:
            s = re.sub(r'[{}]+'.format(punctuation), ' ', s)
        s = [w for w in jieba.cut(s) if w.strip()]
        review_split.append(s)
        words.extend(s)

    vocab = dict(Counter(words).most_common(vocab_size))

    word_to_id = {'UNK': -1, 'PAD': -2}
    id_to_word = {-1: 'UNK', 'PAD': -2}
    reserved_len = len(word_to_id)

    for i, k in enumerate(vocab.keys()):
        if i < (vocab_size - reserved_len):
            word_to_id[k] = i
            id_to_word[i] = k

    # convert word to index
    reveiw_index = []
    for r in tqdm(review_split):
        r_index = []
        for w in r:
            if w in word_to_id.keys():
                r_index.append(word_to_id[w])
            else:
                r_index.append(word_to_id['UNK'])
        if padding and len(r_index) < max_sentence_len:
            for i in range(max_sentence_len - len(r_index)):
                r_index.append(word_to_id['PAD'])

        reveiw_index.append(r_index)

    df['review_split'] = review_split
    df['review_index'] = reveiw_index

    return df




if __name__ == '__main__':
    file_path = '../data/waimai_10k.csv'
    # waimai_data, waimai_label = convert_waimai_data_to_uniform_txt_corpus(file_path)
    # print('waimai_data: {}'.format(len(waimai_data)))
    # print('waimai_label: {}'.format(len(waimai_label)))
    # print('X_sample: {}, y_sample: {}'.format(waimai_data[0], waimai_label[0]))
    #
    # max_sentence_len = 15
    # corpus = Corpus(waimai_data, vocab_size=5000)
    # corpus_indexed = corpus.encoding_corpus_to_index(max_sentence_len)
    #
    # print('word frequency: {}'.format(corpus.word_frequency[:5]))
    # print('index: {}'.format(corpus_indexed[0]))
    # print('string: {}'.format([corpus.id_to_word[id] for id in corpus_indexed[0]]))
    waimai_dataset = get_waimai_dataset(file_path, is_remove_punctuation=True, vocab_size=1000,
                                        max_sentence_len=15, padding=True)
    print(waimai_dataset.head())