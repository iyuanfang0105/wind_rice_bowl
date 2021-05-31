# %%
import io
import itertools
import numpy as np
import os
import re
import string
import tensorflow as tf
import tqdm

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Dot, Embedding, Flatten, GlobalAveragePooling1D, Reshape
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

SEED = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE


# %%
class Dataset(object):
    """
    dataset for word2vec, input from a txt file
    """

    def __init__(self, data_path, vocab_size=4096, batch_size=1024, sequence_len=10, window_size=2, num_ns=4):
        self.data_path = data_path
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.window_size = window_size
        self.num_ns = num_ns
        self.vocab = []

    def get_data(self, debug=False):
        """
        get lines from txt file, and filter the empty lines
        :return:
        """
        text_dataset = tf.data.TextLineDataset(self.data_path).filter(lambda x: tf.cast(tf.strings.length(x), bool))
        text_dataset_vectorized = self.vectorize_text(text_dataset, debug=debug)
        targets, contexts, labels = self.generate_training_data(sequences=text_dataset_vectorized,
                                                                window_size=self.window_size,
                                                                num_ns=self.num_ns,
                                                                vocab_size=self.vocab_size)

        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        dataset = dataset.shuffle(buffer_size=10000).batch(self.batch_size, drop_remainder=True)
        dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

        return dataset

    def vectorize_text(self, text_dataset, debug=False):
        """

        :param text_dataset:
        :return:
        """
        vectorize_layer = TextVectorization(standardize=self.custom_standardization,
                                            max_tokens=self.vocab_size,
                                            output_mode='int',
                                            output_sequence_length=self.sequence_len)
        vectorize_layer.adapt(text_dataset.batch(self.batch_size))
        self.vocab = vectorize_layer.get_vocabulary()

        text_vector_ds = text_dataset.batch(self.batch_size).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()

        sequences = list(text_vector_ds.as_numpy_iterator())

        if debug:
            print(f"====>>>> length of sequences: {len(sequences)}")

            for seq in sequences[:5]:
                print(f"====>>>> {seq} => {[self.vocab[i] for i in seq]}")

        return sequences

    @staticmethod
    def custom_standardization(input_data):
        """
        create a custom standardization function to lowercase the text and remove punctuation
        :return: tf.strings
        """
        lowercase = tf.strings.lower(input_data)
        return tf.strings.regex_replace(lowercase,
                                        '[%s]' % re.escape(string.punctuation), '')

    @staticmethod
    def generate_training_data(sequences, window_size, num_ns, vocab_size, seed=42):
        # Elements of each training example are appended to these lists.
        targets, contexts, labels = [], [], []

        # Build the sampling table for vocab_size tokens.
        sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

        # Iterate over all sequences (sentences) in dataset.
        for sequence in tqdm.tqdm(sequences):

            # Generate positive skip-gram pairs for a sequence (sentence).
            positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
                sequence,
                vocabulary_size=vocab_size,
                sampling_table=sampling_table,
                window_size=window_size,
                negative_samples=0)

            # Iterate over each positive skip-gram pair to produce training examples
            # with positive context word and negative samples.
            for target_word, context_word in positive_skip_grams:
                context_class = tf.expand_dims(
                    tf.constant([context_word], dtype="int64"), 1)
                negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                    true_classes=context_class,
                    num_true=1,
                    num_sampled=num_ns,
                    unique=True,
                    range_max=vocab_size,
                    seed=SEED,
                    name="negative_sampling")

                # Build context and label vectors (for one target word)
                negative_sampling_candidates = tf.expand_dims(
                    negative_sampling_candidates, 1)

                context = tf.concat([context_class, negative_sampling_candidates], 0)
                label = tf.constant([1] + [0] * num_ns, dtype="int64")

                # Append each element from the training example to global lists.
                targets.append(target_word)
                contexts.append(context)
                labels.append(label)
        return targets, contexts, labels


class Word2Vec(Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(Word2Vec, self).__init__()
        self.target_embedding = Embedding(vocab_size,
                                          embedding_dim,
                                          input_length=1,
                                          name="w2v_embedding", )
        self.context_embedding = Embedding(vocab_size,
                                           embedding_dim,
                                           input_length=num_ns + 1)
        self.dots = Dot(axes=(3, 2))
        self.flatten = Flatten()

    def call(self, pair):
        target, context = pair
        we = self.target_embedding(target)
        ce = self.context_embedding(context)
        dots = self.dots([ce, we])
        return self.flatten(dots)

    # @staticmethod
    # def custom_loss(x_logit, y_true):
    #     return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)


# %%
file_path = 'data/poetry/tang.txt'
vocab_size = 3500
batch_size = 1024
sequence_len = 50
window_size = 2
num_ns = 4
embedding_dim = 128

ds = Dataset(file_path,
             vocab_size=vocab_size, batch_size=batch_size, sequence_len=sequence_len,
             window_size=window_size, num_ns=num_ns)
dataset = ds.get_data(debug=True)

model = Word2Vec(vocab_size=vocab_size, embedding_dim=embedding_dim, num_ns=num_ns)
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(dataset, epochs=20)

# %%
weights = model.get_layer('w2v_embedding').get_weights()[0]
vocab = ds.vocab

# %%
out_v = io.open('data/poetry/tang_vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('data/poetry/tang_metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
    if index == 0: continue  # skip 0, it's padding.
    vec = weights[index]
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_m.write(word + "\n")
    
out_v.close()
out_m.close()
