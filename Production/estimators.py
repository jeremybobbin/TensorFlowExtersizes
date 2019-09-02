#!/bin/python
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import subprocess
import sys

import functools
import matplotlib.pyplot as plt
import pandas
import tensorflow as tf
import tensorflow.feature_column as fc

from IPython.display import clear_output

models_path = os.path.join(os.getcwd(), 'models')
sys.path.append(models_path)

 # TF docs may be old... mentioned official.wide_deep
from official.r1.wide_deep import census_dataset
from official.r1.wide_deep import census_main

def view_plot(plt, viewer='feh'):
    child = subprocess.Popen([viewer, '-'], stdin=subprocess.PIPE)
    plt.savefig(child.stdin, format='png')
    return child

def easy_input_function(df, label_key, num_epochs, shuffle, batch_size):
  label = df[label_key]
  ds = tf.data.Dataset.from_tensor_slices((dict(df),label))

  if shuffle:
    ds = ds.shuffle(10000)

  ds = ds.batch(batch_size).repeat(num_epochs)

  return ds


census_dataset.download("/tmp/census_data/")

if "PYTHONPATH" in os.environ:
  os.environ['PYTHONPATH'] += os.pathsep +  models_path
else:
  os.environ['PYTHONPATH'] = models_path

train_file = "/tmp/census_data/adult.data"
test_file = "/tmp/census_data/adult.test"

train_df = pandas.read_csv(train_file, header = None, names = census_dataset._CSV_COLUMNS)
test_df = pandas.read_csv(test_file, header = None, names = census_dataset._CSV_COLUMNS)

train_df.head()

ds = easy_input_function(train_df, label_key='income_bracket', num_epochs=5, shuffle=True, batch_size=10)

# categorical | continuous
for feature_batch, label_batch in ds.take(1):
  print('Feature keys:', list(feature_batch.keys())[:5])
  print()
  print('Age batch   :', feature_batch['age'])
  print()
  print('Label batch :', label_batch )


train_inpf = functools.partial(census_dataset.input_fn, train_file, num_epochs=2, shuffle=True, batch_size=64)
test_inpf = functools.partial(census_dataset.input_fn, test_file, num_epochs=1, shuffle=False, batch_size=64)

age = fc.numeric_column('age')
# left off at: https://www.tensorflow.org/tutorials/estimators/linear#base_feature_columns
