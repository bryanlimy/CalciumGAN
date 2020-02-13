import numpy as np
import tensorflow as tf


def count_trainable_params(model):
  ''' return the number of trainable parameters'''
  return np.sum(
      [tf.keras.backend.count_params(p) for p in model.trainable_weights])


def calculate_convolution_steps(layer, output, kernel_size, strides, padding):
  ''' Calculate the amount of steps in the input is needed for Conv1DTranpose 
  to return a tensor with output number of steps
  '''
  steps = (1 / strides) * (output + 2 * padding - kernel_size) + 1
  if layer > 1:
    steps = calculate_convolution_steps(
        layer=layer - 1,
        output=steps,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding)
  if not steps.is_integer():
    raise ValueError('Conv1D: step {} is not an integer.'.format(steps))
  return steps


def calculate_input_config(num_neurons,
                           noise_dim,
                           num_convolution=0,
                           kernel_size=0,
                           strides=0,
                           padding=0):
  if num_convolution == 0:
    num_units = num_neurons
  else:
    num_units = calculate_convolution_steps(
        layer=num_convolution,
        output=num_neurons,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding)
  return (int(num_units), noise_dim), int(num_units) * noise_dim


class Conv1DTranspose(tf.keras.layers.Layer):

  def __init__(
      self,
      filters,
      kernel_size,
      strides,
      padding='valid',
      output_padding=None,
      activation=0,
  ):
    super().__init__()
    self.activation = tf.keras.layers.Activation(
        activation) if activation else None

    self._conv2dtranspose = tf.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=(kernel_size, 1),
        strides=(strides, 1),
        padding=padding,
        output_padding=output_padding)

  def call(self, inputs):
    outputs = tf.expand_dims(inputs, axis=2)
    outputs = self._conv2dtranspose(outputs)
    outputs = tf.squeeze(outputs, axis=2)

    if self.activation is not None:
      outputs = self.activation(outputs)

    return outputs
