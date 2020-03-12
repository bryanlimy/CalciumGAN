import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def activation_fn(name, **kwargs):
  return layers.LeakyReLU() if name == 'leakyrelu' else layers.Activation(
      name, **kwargs)


def count_trainable_params(model):
  ''' return the number of trainable parameters'''
  return np.sum(
      [tf.keras.backend.count_params(p) for p in model.trainable_weights])


def calculate_convolution_steps(layer, output, kernel_size, strides, padding):
  '''
  Calculate the amount of steps in the input is needed for Conv1DTranpose 
  to return a tensor with output number of steps
  :param layer: current layer index
  :param output: size of output
  :param kernel_size: kernel size
  :param strides: stride size
  :param padding: type of padding
  :return: the steps size to Conv1DTranpose to get output size
  '''
  if padding == 'same':
    steps = output / strides
  else:
    steps = (1 / strides) * (output - kernel_size) + 1

  if not steps.is_integer():
    raise ValueError('Conv1D: step {} is not an integer.'.format(steps))

  if layer > 1:
    steps = calculate_convolution_steps(
        layer=layer - 1,
        output=steps,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding)

  return steps


def calculate_input_config(output,
                           noise_dim,
                           num_convolution=0,
                           kernel_size=0,
                           strides=0,
                           padding='same'):
  if num_convolution == 0:
    num_units = output
  else:
    num_units = calculate_convolution_steps(
        layer=num_convolution,
        output=output,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding)
  return (int(num_units), noise_dim), int(num_units) * noise_dim


class Conv1DTranspose(layers.Layer):

  def __init__(
      self,
      filters,
      kernel_size,
      strides,
      padding='same',
      output_padding=None,
      activation='linear',
  ):
    super().__init__()
    self.activation = activation_fn(activation)

    self._conv2dtranspose = layers.Conv2DTranspose(
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
