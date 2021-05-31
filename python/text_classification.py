import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jieba
import argparse

from collections import Counter

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Bidirectional
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence
from keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy


punctuations = "，。！……？（）【】~!@#$%^&*()_+`{}|\[\]\:\";\-\\\='<>?,./"


def get_imdb_dataset(num_words=1000, max_len=50):
    # load the dataset
    NUM_WORDS = num_words  # only use top 1000 words
    INDEX_FROM = 3  # word index offset

    (train_X, train_y), (test_X, test_y) = imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)

    word_to_id = imdb.get_word_index()
    word_to_id = {k: (v + INDEX_FROM) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    word_to_id["<UNUSED>"] = 3

    id_to_word = {value: key for key, value in word_to_id.items()}

    train_X = sequence.pad_sequences(train_X, maxlen=max_len)
    test_X = sequence.pad_sequences(test_X, maxlen=max_len)

    return train_X, train_y, test_X, test_y, word_to_id, id_to_word


def simple_rnn(vocab_dim, embedding_dim, hidden_dim):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_dim, output_dim=embedding_dim))
    model.add(SimpleRNN(hidden_dim))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def get_weibo_comment_data(data_path, remove_puncts=True, vocab_size=1000, sentence_max_len=50, debug=False):
    # Load the raw data from csv
    pd_all = pd.read_csv(data_path)
    print('====>>>> Comments Num: {}'.format(pd_all.shape[0]))
    print('====>>>> Positive: {}'.format(pd_all[pd_all.label == 1].shape[0]))
    print('====>>>> Negative: {}'.format(pd_all[pd_all.label == 0].shape[0]))
    print(pd_all.sample(5))

    if debug:
        pd_all = pd_all.sample(150)

    # Refine the data
    comments = [] # remove the punctuations
    comments_splited = [] # splitting
    labels = [] # label for classification
    words = []

    for index, row in pd_all.iterrows():
        if row.shape[0] == 2:
            row = row.values
            comment = row[1].strip()

            labels.append(row[0])

            if remove_puncts:
                comment = remove_punctuations(comment)

            comment_words = splitting_sentence(comment)

            words += comment_words
            comments.append(comment)
            comments_splited.append(splitting_sentence(comment))

    # Build Vocab
    id_to_word, word_to_id = build_vocab(words, vocab_size=vocab_size)

    # Indexing the text
    comments_indexing = []
    for cmt in comments_splited:
        cmt_inds = sentence_to_ids(cmt, word_to_id, max_len=sentence_max_len)
        comments_indexing.append(cmt_inds)

    data_df = pd.DataFrame({'comment': comments,
                            'comment_splitted': comments_splited,
                            'comment_indexing': comments_indexing,
                            'label': labels})
    print(data_df.sample(5))

    return data_df, id_to_word, word_to_id


def build_corpus():
    return 0


def remove_punctuations(s):
    punctuations = "，。！……？（）【】~!@#$%^&*()_+`{}|\[\]\:\";\-\\\='<>?,./"
    s = re.sub(r'[{}]+'.format(punctuations), ' ', s)
    return s


def splitting_sentence(s):
    tmp = []
    for w in jieba.cut(s):
        if w.strip():
            tmp.append(w.strip())
    return tmp


def build_vocab(words, vocab_size=1000):
    vocab = Counter(words).most_common(vocab_size)
    word_to_id = {'UNK': 0, 'START': 1, 'END': 2, 'PADDING': 3}
    id_to_word = {0: 'UNK', 1: 'START', 2:'END', 3: 'PADDING'}

    init_index = len(word_to_id)

    for idx, v in enumerate(vocab):
        word_to_id[v[0]] = idx + init_index
        id_to_word[idx + init_index] = v[0]

    return id_to_word, word_to_id


def sentence_to_ids(s, word_to_id, max_len=20):
    s_inds = [word_to_id['START']]
    for w in s:
        if w in word_to_id.keys():
            s_inds.append(word_to_id[w])
        else:
            s_inds.append(word_to_id['UNK'])
    if len(s_inds) < max_len-1:
        for i in range(max_len - 1 - len(s_inds)):
            s_inds.append(word_to_id['PADDING'])
    else:
        s_inds = s_inds[0:max_len-1]

    s_inds.append(word_to_id['END'])

    assert len(s_inds) == max_len, print('====>>>> Do not match max_len of sentence, length: {}'.format(len(s_inds)))
    return s_inds


def ids_to_sentence(s, id_to_word, UNK=-1):
    return 0


def build_model(vocab_size, embedding_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2))

    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path',
                        required=False,
                        type=str,
                        default='~/WORK/open_source_proj/ChineseNlpCorpus-master/datasets/weibo_senti_100k/weibo_senti_100k.csv',
                        help='Please input the corpus data path')

    parser.add_argument('-debug',
                        required=False,
                        default=False,
                        action='store_true',
                        help='Flag of debugging')

    parser.add_argument('-epochs',
                        required=False,
                        type=int,
                        default=10,
                        help='Please input training epochs')


    args = parser.parse_args()

    print(args)

    weibo_comments, id_to_word, word_to_id = get_weibo_comment_data(args.data_path,
                                                                    remove_puncts=True,
                                                                    vocab_size=100,
                                                                    sentence_max_len=20,
                                                                    debug=args.debug)

    X = np.vstack(weibo_comments['comment_indexing'].values)
    y = np.vstack(weibo_comments['label'].values)
    print('X: {}, y: {}'.format(X.shape, y.shape))

    vocab_size = len(word_to_id)
    embedding_size = 300

    model = build_model(vocab_size, embedding_size)
    print(model.summary())

    model.fit(X, y, epochs=args.epochs)
    print()
