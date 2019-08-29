#!/bin/python


# For reference
#
#class DenseLayer(tf.keras.layers.Layer):
#  def __init__(self, num_outputs):
#    super(DenseLayer, self).__init__() #???
#    self.num_outputs = num_outputs
#
#  def build(self, input_shape):
#    self.kernel = self.add_variable("kernel",
#                                    shape=[int(input_shape[-1]),
#                                           self.num_outputs])
#
#  def call(self, input):
#    return tf.matmul(input, self.kernel)
# 
class Animal:
  def __init__(self, name):
      self.name = name

  def get_name(self):
      return self.name
  
  def breath(self):
      name = self.name
      print("{} inhales... {} exhales".format(name, name))

class Reptile(Animal):
  def __init__(self, name, has_scales):
      super().__init__(name)
      self.has_scales = has_scales

  def breath(self):
      noise = ""
      for _ in range(10):
          noise += 's'

      print(noise)

hippo = Animal("Hippo")
snake = Reptile("Snake", True)

hippo.breath()
snake.breath()
