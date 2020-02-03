import numpy as np
import tensorflow as tf


def kl_divergence(real, fake):
  return tf.reduce_mean(tf.keras.losses.KLD(y_true=real, y_pred=fake))


def min_signals_error(real, fake):
  return tf.reduce_mean(
      tf.square(tf.reduce_min(real, axis=-1) - tf.reduce_min(fake, axis=-1)))


def max_signals_error(real, fake):
  return tf.reduce_mean(
      tf.square(tf.reduce_max(real, axis=-1) - tf.reduce_max(fake, axis=-1)))


def mean_signals_error(real, fake):
  return tf.reduce_mean(
      tf.square(tf.reduce_mean(real, axis=-1) - tf.reduce_mean(fake, axis=-1)))


def std_signals_error(real, fake):
  return tf.reduce_mean(
      tf.square(
          tf.math.reduce_std(real, axis=-1) -
          tf.math.reduce_std(fake, axis=-1)))


def cross_correlation(real, fake):
  shape = (real.shape[0] * real.shape[1], real.shape[2])
  real = tf.reshape(real, shape=shape)
  fake = tf.reshape(fake, shape=shape)

  def _cross_correlation(x, y):
    _results = np.corrcoef(x, y)
    _results = np.diagonal(_results, offset=len(x))
    return _results

  results = tf.py_function(
      _cross_correlation, inp=[real, fake], Tout=tf.float32)

  return tf.reduce_mean(results)
