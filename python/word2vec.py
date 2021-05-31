import tensorflow as tf
from tensorflow import keras

max_features = 1000
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=max_features)

maxlen = 100
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

vocb_size = max_features
embedding_dim = 300
model = keras.Sequential([
    keras.layers.Embedding(vocb_size, embedding_dim),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))
print(model.summary())

embeddings_layer = model.layers[0]
embeddings_weights = embeddings_layer.get_weights()[0]
print('===>>> shape of embeddings: {}'.format(embeddings_weights.shape))
