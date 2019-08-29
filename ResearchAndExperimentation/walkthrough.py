#!/bin/python
from __future__ import absolute_import, division, print_function, unicode_literals

import io
import math
import os
import subprocess

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.compat.v1.enable_eager_execution()

def view_plot(plt, viewer='feh'):
    child = subprocess.Popen([viewer, '-'], stdin=subprocess.PIPE)
    plt.savefig(child.stdin, format='png')
    return child

def pack_features_vector(features, labels):
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url), # <-- neat use of basename
                                           origin=train_dataset_url)
print("Local copy of the dataset file: {}".format(train_dataset_fp))

subprocess.Popen(["head", "-n5", train_dataset_fp], shell=False)

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

batch_size = 32

train_dataset = tf.contrib.data.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

features, labels = next(iter(train_dataset))
# features [[120], [4], [setosa], [versicolor]]
# labels [Enum species]

for value in features.values():
    print("Value: {}".format(value))

train_dataset = train_dataset.map(pack_features_vector)

print(features)
print(labels)

# left off at https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#select_the_model
