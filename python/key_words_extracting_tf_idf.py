import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

train_data_path = '/Users/wind/WORK/open_source_proj/nlp-in-practice-master/tf-idf/data/stackoverflow-data-idf.json'
test_data_path = '/Users/wind/WORK/open_source_proj/nlp-in-practice-master/tf-idf/data/stackoverflow-test.json'
stop_words_file = '/Users/wind/WORK/open_source_proj/nlp-in-practice-master/tf-idf/resources/stopwords.txt'

data_df = pd.read_json(train_data_path, lines=True)
print('\n====>>>> data types:\n {}'.format(data_df.dtypes))
print('\n====>>>> data shape:\n {}'.format(data_df.shape))


def pre_process(text):
    # lowercase
    text = text.lower()

    # remove tags
    text = re.sub("</?.*?>", " <> ", text)

    # remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)

    return text


data_df['text'] = data_df['title'] + data_df['body']
data_df['text'] = data_df['text'].apply(lambda x: pre_process(x))

# show the first 'text'
print(data_df['text'][2])


def get_stop_words(stop_file_path):
    """load stop words """

    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)


# load a set of stop words
stopwords = get_stop_words(stop_words_file)

# get the text column
docs = data_df['text'].tolist()

# create a vocabulary of words,
# ignore words that appear in 85% of documents,
# eliminate stop words

cv = CountVectorizer(max_df=0.85, stop_words=stopwords, max_features=100)
word_count_vector = cv.fit_transform(docs)
print('\n====>>>> shape of word count matrix: {}\n'.format(word_count_vector.shape))
print('\n====>>>> key words of docs: {}'.format(cv.get_feature_names()))

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)

# read test docs into a dataframe and concatenate title and body
df_test = pd.read_json(test_data_path, lines=True)
df_test['text'] = df_test['title'] + df_test['body']
df_test['text'] = df_test['text'].apply(lambda x: pre_process(x))

# get test docs into a list
docs_test = df_test['text'].tolist()
docs_title = df_test['title'].tolist()
docs_body = df_test['body'].tolist()


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]

        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


# you only needs to do this once
feature_names = cv.get_feature_names()

# get the document that we want to extract keywords from
doc = docs_test[0]

# generate tf-idf for the given document
tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))

# sort the tf-idf vectors by descending order of scores
sorted_items = sort_coo(tf_idf_vector.tocoo())

# extract only the top n; n here is 10
keywords = extract_topn_from_vector(feature_names, sorted_items, 10)

# now print the results
print("\n=====Title=====")
print(docs_title[0])
print("\n=====Body=====")
print(docs_body[0])
print("\n===Keywords===")
for k in keywords:
    print(k, keywords[k])



