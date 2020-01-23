from .registry import losses_register as register

import tensorflow as tf


@register('gan')
def get_losses(hparams):
  cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  def _generator_loss(real_output, fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

  def _discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    penalty = 0.0
    return real_loss + fake_loss, penalty

  return _generator_loss, _discriminator_loss
