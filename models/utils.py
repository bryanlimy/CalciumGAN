import tensorflow as tf


def get_activation_fn(activation):
  return tf.keras.activations.get(activation)


class Conv1DTranspose(tf.keras.layers.Layer):

  def __init__(
      self,
      filters,
      kernel_size,
      strides,
      padding='valid',
      activation=0,
  ):
    super().__init__()
    self.activation = get_activation_fn(activation) if activation else None

    self.conv2dtranspose = tf.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=(kernel_size, 1),
        strides=(strides, 1),
        padding=padding)

  def call(self, inputs):
    outputs = tf.expand_dims(inputs, axis=2)
    outputs = self.conv2dtranspose(outputs)
    outputs = tf.squeeze(outputs, axis=2)

    if self.activation is not None:
      outputs = self.activation(outputs)

    return outputs
