#!/bin/python
from __future__ import absolute_import, division, print_function, unicode_literals

# std
import subprocess


import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras

def multi_hot_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0 # reeeee
    return results

def view_plot(plt, viewer='feh'):
    child = subprocess.Popen([viewer, '-'], stdin=subprocess.PIPE)
    plt.savefig(child.stdin, format='png')
    return child

def sized_model(size, train_data, train_labels, test_data, test_labels):
    model = keras.Sequential([
        keras.layers.Dense(size, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
        keras.layers.Dense(size, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy', 'binary_crossentropy'])

    history = model.fit(train_data,
                                          train_labels,
                                          epochs=20,
                                          batch_size=512,
                                          validation_data=(test_data, test_labels),
                                          verbose=2)

    return model, history

def plot_history(histories, key='binary_crossentropy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])


NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

(base_model, base_hist) = sized_model(16, train_data, train_labels, test_data, test_labels);
(small_model, small_hist) = sized_model(4, train_data, train_labels, test_data, test_labels);
(big_model, big_hist) = sized_model(512, train_data, train_labels, test_data, test_labels);




plot_history([('baseline', base_hist),
              ('smaller', small_hist),
              ('bigger', big_hist)])
