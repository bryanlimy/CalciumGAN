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

  def random_weighted_average(self, inputs, fake):
    alpha = tf.random.uniform((inputs.shape[0], 1, 1))
    return (alpha * inputs) + ((1 - alpha) * fake)

  @tf.function
  def gradient_penalty(self, real, fake):
    interpolated = self.random_weighted_average(real, fake)
    interpolated_output = self.discriminator(interpolated, training=True)
    gradients = tf.gradients(interpolated_output, interpolated)
    gradients_sqr = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(
        gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradients_l2_norm = tf.sqrt(gradients_sqr_sum)
    gradient_penalty = tf.square(gradients_l2_norm)
    return tf.reduce_mean(gradient_penalty)

  def discriminator_loss(self, real_output, fake_output, real=None, fake=None):
    real_loss = -tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    gradient_penalty = self.gradient_penalty(real, fake)
    loss = real_loss + fake_loss + self._lambda * gradient_penalty
    return loss, gradient_penalty

  @tf.function
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

  @tf.function
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

  def train(self, inputs):
    dis_losses, gradient_penalties = [], []
    for i in range(self._n_critic):
      dis_loss, gradient_penalty = self._train_discriminator(inputs)
      dis_losses.append(dis_loss)
      gradient_penalties.append(gradient_penalty)

    gen_loss, kl = self._train_generator(inputs)

    return gen_loss, tf.reduce_mean(dis_losses), tf.reduce_mean(
        gradient_penalties), kl
