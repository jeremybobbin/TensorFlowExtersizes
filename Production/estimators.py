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
train_inpf = functools.partial(census_dataset.input_fn, train_file, num_epochs=2, shuffle=True, batch_size=64)
test_inpf = functools.partial(census_dataset.input_fn, test_file, num_epochs=1, shuffle=False, batch_size=64)

age = fc.numeric_column('age')
classifier = tf.estimator.LinearClassifier(feature_columns=[age])
classifier.train(train_inpf)
result = classifier.evaluate(test_inpf)

clear_output()
# accuracy
#	Percentage of correct number of classifications
#
# accuracy_baseline
#	Accuracy baseline based on labels mean. This is the best the model could do by always predicting one class.
#
# AUC or Area Under the (ROC) Curve
#	tells you something about the true/false positive rate.
#	In short the AUC is equal to the probability that a classifier will rank a randomly chosen positive instance higher than a randomly chosen negative one.
#       
#       https://www.tensorflow.org/api_docs/python/tf/metrics/auc
#	https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
#	https://developers.google.com/machine-learning/crash-course/classification/true-false-positive-negative
#   
# auc_precision_recall
# 	Is the percentage of relevant intstances, among the retrieved instances, that have been retrieved over the total amount of relevant instances.
#
# average_loss
# 	You're usually minimizing some function, and this is likely the average value of that function given the current batches.
#
# loss
# 	The current value of the loss (as above). Either the sum of the losses, or the loss of the last batch.
#
# global_step
# 	Number of iterations.
#
# label/mean and prediction/mean

print(result)
