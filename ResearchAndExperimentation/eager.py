#!/bin/python

import time
import tempfile
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

x = tf.random_uniform([3, 3])

print(x.device) # :-) /job:localhost/replica:0/task:0/device:CPU:0 
print(tf.test.is_gpu_available())

def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.matmul(x, x)

  result = time.time()-start

  print("10 loops: {:0.2f}ms".format(1000*result))


# Force execution on CPU
print("On CPU:")

if False: 
    with tf.device("CPU:0"):
      x = tf.random_uniform([1000, 1000])
      assert x.device.endswith("CPU:0")
      time_matmul(x) # 10 loops: 389.61ms

if tf.test.is_gpu_available():
  with tf.device("GPU:0"):
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)


ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
  f.write("""Line 1
Line 2
Line 3
Line 4
Line 5
Line 6
Line 7
Line 8
Line 9
Line 0
  """)

ds_file = tf.data.TextLineDataset(filename)

ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

ds_file = ds_file.batch(2)

print('Elements of ds_tensors:')
for x in ds_tensors:
  print(x)

print('\nElements in ds_file:')
for x in ds_file:
  print(x)
