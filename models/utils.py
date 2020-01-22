import tensorflow as tf


def get_activation_fn(activation):
  return tf.keras.activations.get(activation)
