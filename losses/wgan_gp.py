from .registry import losses_register as register

import numpy as np
import tensorflow as tf


@register('wgan-gp')
def get_losses(hparams):

  def _generator_loss(real_output, fake_output):
    return -tf.reduce_mean(fake_output)

  def _random_weighted_average(inputs):
    alpha = tf.random.uniform((inputs[0].shape[0], 1, 1))
    return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

  def _gradient_penalty(prediction, average):
    gradients = tf.gradients(prediction, average)[0]
    gradients_sqr = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(
        gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradients_l2_norm = tf.sqrt(gradients_sqr_sum)
    gradient_penalty = tf.square(gradients_l2_norm)
    return tf.reduce_mean(gradient_penalty)

  def _discriminator_loss(real_output, fake_output, discriminator, inputs,
                          generated):
    real_loss = -tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)

    interpolated_samples = _random_weighted_average([inputs, generated])
    validity_interpolated = discriminator(interpolated_samples, training=True)

    gradient_penalty = _gradient_penalty(validity_interpolated,
                                         interpolated_samples)
    total_loss = real_loss + fake_loss + hparams.gradient_penalty * gradient_penalty
    return total_loss, gradient_penalty

  return _generator_loss, _discriminator_loss
