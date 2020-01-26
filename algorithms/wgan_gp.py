from .registry import register

import numpy as np
import tensorflow as tf

from .gan import GAN


@register('wgan-gp')
class WGAN_GP(GAN):

  def __init__(self, hparams, generator, discriminator, summary):
    super().__init__(hparams, generator, discriminator, summary)

    self._lambda = hparams.gradient_penalty
    self._n_critic = hparams.n_critic

  def generator_loss(self, fake_output):
    return -tf.reduce_mean(fake_output)

  def interpolation(self, real, fake):
    alpha = tf.random.uniform((real.shape[0], 1, 1), minval=0.0, maxval=1.0)
    return (alpha * real) + ((1 - alpha) * fake)

  def gradient_penalty(self, real, fake, training=True):
    interpolated = self.interpolation(real, fake)
    with tf.GradientTape() as tape:
      tape.watch(interpolated)
      interpolated_output = self.discriminator(interpolated, training=training)
    gradient = tape.gradient(interpolated_output, interpolated)
    norm = tf.norm(tf.reshape(gradient, shape=(gradient.shape[0], -1)), axis=1)
    return tf.reduce_mean(tf.square(norm - 1.0))

  def discriminator_loss(self,
                         real_output,
                         fake_output,
                         real=None,
                         fake=None,
                         training=True):
    real_loss = -tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    gradient_penalty = self.gradient_penalty(real, fake, training=training)
    loss = real_loss + fake_loss + self._lambda * gradient_penalty
    return loss, gradient_penalty

  def _train_generator(self, inputs):
    noise = tf.random.normal((inputs.shape[0], self._num_neurons,
                              self._noise_dim))

    with tf.GradientTape() as tape:
      fake = self.generator(noise, training=True)
      fake_output = self.discriminator(fake, training=True)

      gen_loss = self.generator_loss(fake_output)

    gen_gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
    self.gen_optimizer.apply_gradients(
        zip(gen_gradients, self.generator.trainable_variables))

    kl = self.kl_divergence(real=inputs, fake=fake)

    return gen_loss, kl

  def _train_discriminator(self, inputs):
    noise = tf.random.normal((inputs.shape[0], self._num_neurons,
                              self._noise_dim))

    with tf.GradientTape() as tape:
      fake = self.generator(noise, training=True)

      real_output = self.discriminator(inputs, training=True)
      fake_output = self.discriminator(fake, training=True)

      dis_loss, gradient_penalty = self.discriminator_loss(
          real_output, fake_output, real=inputs, fake=fake)

    dis_gradients = tape.gradient(dis_loss,
                                  self.discriminator.trainable_variables)
    self.dis_optimizer.apply_gradients(
        zip(dis_gradients, self.discriminator.trainable_variables))

    return dis_loss, gradient_penalty

  #@tf.function
  def train(self, inputs):
    dis_losses, gradient_penalties = [], []
    for i in range(self._n_critic):
      dis_loss, gradient_penalty = self._train_discriminator(inputs)
      dis_losses.append(dis_loss)
      gradient_penalties.append(gradient_penalty)

    gen_loss, kl = self._train_generator(inputs)

    return gen_loss, tf.reduce_mean(dis_losses), tf.reduce_mean(
        gradient_penalties), kl
