#!/bin/python
from __future__ import absolute_import, division, print_function, unicode_literals

import subprocess
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()

def loss(predicted_y, desired_y):
  return tf.reduce_mean(tf.square(predicted_y - desired_y))

def train(model, inputs, outputs, learning_rate):
  with tf.GradientTape() as t:
    current_loss = loss(model(inputs), outputs)

  dW, db = t.gradient(current_loss, [model.W, model.b])
  model.W.assign_sub(learning_rate * dW)
  model.b.assign_sub(learning_rate * db)

class Model(object):
  def __init__(self):
    self.W = tf.Variable(5.0)
    self.b = tf.Variable(0.0)

  # wow great class
  def __call__(self, x):
    return self.W * x + self.b

def view_plot(plt, viewer='feh'):
    child = subprocess.Popen([viewer, '-'], stdin=subprocess.PIPE)
    plt.savefig(child.stdin, format='png')
    return child

model = Model()

TRUE_W = 5.0
TRUE_b = 2.1
NUM_EXAMPLES = 1000

inputs  = tf.random_normal(shape=[NUM_EXAMPLES])
noise   = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

model = Model()

Ws, bs = [], []
epochs = range(500)

for epoch in epochs:
  Ws.append(model.W.numpy())
  bs.append(model.b.numpy())
  current_loss = loss(model(inputs), outputs)

  train(model, inputs, outputs, learning_rate=0.1)
  print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
        (epoch, Ws[-1], bs[-1], current_loss))

plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true_b'])

view_plot(plt)
