#!/bin/python
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tf.enable_eager_execution()

class DenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(DenseLayer, self).__init__() #???
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_variable("kernel",
                                    shape=[int(input_shape[-1]),
                                           self.num_outputs])

  def call(self, input):
    return tf.matmul(input, self.kernel)

layer = DenseLayer(10)
print(layer(tf.zeros([10, 5])))
print(layer.trainable_variables)
