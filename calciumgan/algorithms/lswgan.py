from .registry import register

import numpy as np
import tensorflow as tf

from .gan import GAN


@register('lswgan')
class LSWGAN(GAN):

  def __init__(self, hparams, generator, discriminator, summary):
    super().__init__(hparams, generator, discriminator, summary)

    self.penalty = hparams.gradient_penalty
    self.n_critic = hparams.n_critic

  def generator_loss(self, fake_output):
    return -tf.reduce_mean((1.0 - fake_output)**2)

  def _train_generator(self, inputs):
    noise = self.get_noise(batch_size=inputs.shape[0])

    with tf.GradientTape() as tape:
      fake = self.generator(noise, training=True)
      fake_output = self.discriminator(fake, training=True)

      gen_loss = self.generator_loss(fake_output)
      scaled_loss = self.gen_optimizer.get_scaled_loss(gen_loss)

    self.gen_optimizer.update(self.generator, scaled_loss, tape)

    metrics = self.metrics(real=inputs, fake=fake)

    return gen_loss, metrics

  @staticmethod
  def interpolation(real, fake):
    alpha = tf.random.uniform((real.shape[0], 1, 1), minval=0.0, maxval=1.0)
    return (alpha * real) + ((1 - alpha) * fake)

  def gradient_penalty(self, real, fake, training=True):
    interpolated = self.interpolation(real, fake)
    with tf.GradientTape() as tape:
      tape.watch(interpolated)
      interpolated_output = self.discriminator(interpolated, training=training)
    gradient = tape.gradient(interpolated_output, interpolated)
    norm = tf.norm(tf.reshape(gradient, shape=(gradient.shape[0], -1)), axis=1)
    return tf.reduce_mean((1.0 - norm)**2)

  def discriminator_loss(self,
                         real_output,
                         fake_output,
                         real=None,
                         fake=None,
                         training=True):
    real_loss = tf.reduce_mean(real_output**2)
    fake_loss = tf.reduce_mean((1.0 - fake_output)**2)
    gradient_penalty = self.gradient_penalty(real, fake, training=training)
    loss = real_loss + fake_loss + self.penalty * gradient_penalty
    return loss, gradient_penalty

  def _train_discriminator(self, inputs):
    noise = self.get_noise(batch_size=inputs.shape[0])

    with tf.GradientTape() as tape:
      fake = self.generator(noise, training=True)

      real_output = self.discriminator(inputs, training=True)
      fake_output = self.discriminator(fake, training=True)

      dis_loss, gradient_penalty = self.discriminator_loss(
          real_output, fake_output, real=inputs, fake=fake, training=True)

      scaled_loss = self.dis_optimizer.get_scaled_loss(dis_loss)

    self.dis_optimizer.update(self.discriminator, scaled_loss, tape)

    return dis_loss, gradient_penalty

  @tf.function
  def train(self, inputs):
    dis_losses, gradient_penalties = [], []
    for i in range(self.n_critic):
      dis_loss, gradient_penalty = self._train_discriminator(inputs)
      dis_losses.append(dis_loss)
      gradient_penalties.append(gradient_penalty)

    gen_loss, metrics = self._train_generator(inputs)

    dis_loss = tf.reduce_mean(dis_loss)
    gradient_penalty = tf.reduce_mean(gradient_penalties)

    return gen_loss, dis_loss, gradient_penalty, metrics
