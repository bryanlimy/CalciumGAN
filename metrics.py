import numpy as np
import tensorflow as tf


def get_mean_spike(spikes):
  binarized = tf.cast(
      spikes > tf.random.uniform(spikes.shape), dtype=tf.float32)
  return tf.reduce_mean(binarized)
