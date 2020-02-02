import tensorflow as tf


def get_activation_fn(activation):
  return tf.keras.activations.get(activation)


def calculate_input_steps(output, kernel_size, strides):
  ''' Calculate the amount of steps in the input is needed for Conv1DTranpose 
  to return a tensor with output number of steps
  '''
  steps = (1 / strides) * (output - kernel_size) + 1
  if not steps.is_integer():
    raise ValueError(
        'Conv1D: step {} is not an integer, check model configuration.'.format(
            steps))
  return int(steps)


def calculate_input_config(num_neurons,
                           noise_dim,
                           conv_layers=0,
                           kernel_size=0,
                           strides=0):
  if conv_layers == 0:
    num_units = num_neurons
  else:
    num_units = [num_neurons] + [0] * conv_layers
    for i in range(1, conv_layers + 1):
      num_units[i] = calculate_input_steps(num_units[i - 1], kernel_size,
                                           strides)
    num_units = num_units[-1]
  return (int(num_units), noise_dim), int(num_units) * noise_dim


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
