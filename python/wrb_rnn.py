# %%
import os
import re
import string
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


# %%
class DatasetImdb(object):
    def __init__(self, data_dir, batch_size=32, max_tokens=10000, sequence_length=200, seed=42):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.sequence_length = sequence_length
        self.vocab = []
        self.seed = seed
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    def load_dataset(self):
        raw_train_ds, raw_val_ds, raw_test_ds = self.load_raw_data()
        max_features = 10000
        sequence_length = 250

        vectorize_layer = TextVectorization(standardize=self.custom_standardization,
                                            max_tokens=self.max_tokens,
                                            output_mode='int',
                                            output_sequence_length=sequence_length)

        # Make a text-only dataset (without labels), then call adapt
        train_text = raw_train_ds.map(lambda x, y: x)
        vectorize_layer.adapt(train_text)
        self.vocab = vectorize_layer.get_vocabulary()

        train_ds = raw_train_ds.map(lambda x, y: (vectorize_layer(tf.expand_dims(x, -1)), y))
        val_ds = raw_val_ds.map(lambda x, y: (vectorize_layer(tf.expand_dims(x, -1)), y))
        test_ds = raw_test_ds.map(lambda x, y: (vectorize_layer(tf.expand_dims(x, -1)), y))

        train_ds = train_ds.cache().prefetch(buffer_size=self.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=self.AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=self.AUTOTUNE)

        return train_ds, val_ds, test_ds

    def load_raw_data(self):
        """

        :return: BatchDataset
        """
        train_dir = os.path.join(dataset_dir, 'train')
        test_dir = os.path.join(dataset_dir, 'test')

        # sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
        # with open(sample_file) as f:
        #     print(f.read())

        raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
            train_dir,
            batch_size=self.batch_size,
            validation_split=0.2,
            subset='training',
            seed=self.seed)

        raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
            train_dir,
            batch_size=self.batch_size,
            validation_split=0.2,
            subset='validation',
            seed=self.seed)

        raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
            test_dir,
            batch_size=self.batch_size)

        return raw_train_ds, raw_val_ds, raw_test_ds

    @staticmethod
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

# def plot_graphs(history, metric):
#     plt.plot(history.history[metric])
#     plt.plot(history.history['val_' + metric], '')
#     plt.xlabel("Epochs")
#     plt.ylabel(metric)
#     plt.legend([metric, 'val_' + metric])


# %%
class WrbRnn(object):
    def __init__(self, embedding_dim=128, batch_size=32, max_features=10000):
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.max_features = max_features

    def embedding_cls(self):
        model = tf.keras.Sequential([
        layers.Embedding(self.max_features + 1, self.embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1)])

        print(f"====>>>>Model summary: \n")
        print(model.summary())

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer='adam',
                      metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

        return model

    def bi_directional_rnn(self):
        model = tf.keras.Sequential([
            layers.Embedding(input_dim=self.max_features + 1, output_dim=self.embedding_dim),
            layers.Bidirectional(tf.keras.layers.LSTM(64)),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        print(f"====>>>>Model summary: \n")
        print(model.summary())

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(1e-4),
                      metrics=['accuracy'])

        return model

    @staticmethod
    def train(model, train_ds, val_ds, epochs=10):
        # epochs = 10
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        return history, model

    @staticmethod
    def test(model, test_ds):
        loss, accuracy = model.evaluate(test_ds)
        return loss, accuracy

    @staticmethod
    def plot(history):
        history_dict = history.history
        print(f"====>>>> history keys: \n {history_dict.keys()}")

        acc = history_dict['binary_accuracy']
        val_acc = history_dict['val_binary_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.subplot(121)
        # "bo" is for "blue dot"
        plt.plot(epochs, loss, 'bo', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(122)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        plt.show()


# %%
dataset_dir = '../data/imdb/aclImdb'
imdb_dataset = DatasetImdb(data_dir=dataset_dir)
train_ds, val_ds, test_ds = imdb_dataset.load_dataset()

# %%
for x, y in train_ds.take(1):
    print(f"====>>>> data example:  x: {x}, \n y: {y}")

# %%
wrb_rnn = WrbRnn()
embd = wrb_rnn.embedding_cls()
bi_directional_rnn = wrb_rnn.bi_directional_rnn()

# %%

epochs = 5
# history, embd = wrb_rnn.train(embd, train_ds, val_ds, epochs=epochs)
# wrb_rnn.plot(history)

# %%
history, bi_directional_rnn = wrb_rnn.train(bi_directional_rnn, train_ds, val_ds, epochs=epochs)
wrb_rnn.plot(history)


# %%
