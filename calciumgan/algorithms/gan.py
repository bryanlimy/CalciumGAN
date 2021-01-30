from .registry import register

import tensorflow as tf

from calciumgan.utils.utils import denormalize
from calciumgan.algorithms.optimizer import Optimizer


@register('gan')
class GAN(object):

  def __init__(self, hparams, G, D):
    self.G = G
    self.D = D

    self.G_optimizer = Optimizer(hparams)
    self.D_optimizer = Optimizer(hparams)

    self.noise_shape = hparams.noise_shape
    self.normalize = hparams.normalize
    self.signals_min = hparams.signals_min
    self.signals_max = hparams.signals_max
    self.mixed_precision = hparams.mixed_precision

    self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  def get_models(self):
    return self.G, self.D

  def sample_noise(self, batch_size):
    """ sample noise from standard normal distribution """
    return tf.random.normal((batch_size,) + self.noise_shape)

  def generator_loss(self, discriminate_fake):
    return self.cross_entropy(
        tf.ones_like(discriminate_fake), discriminate_fake)

  def discriminator_loss(self, discriminate_real, discriminate_fake):
    real_loss = self.cross_entropy(
        tf.ones_like(discriminate_real), discriminate_real)
    fake_loss = self.cross_entropy(
        tf.zeros_like(discriminate_fake), discriminate_fake)
    return real_loss + fake_loss

  @tf.function
  def train(self, inputs):
    result = {}
    noise = self.sample_noise(batch_size=inputs.shape[0])
    with tf.GradientTape(persistent=True) as tape:
      fake = self.G(noise, training=True)
      discriminate_real = self.D(inputs, training=True)
      discriminate_fake = self.D(fake, training=True)
      G_loss = self.generator_loss(discriminate_fake)
      D_loss = self.discriminator_loss(discriminate_real, discriminate_fake)
      result.update({'G_loss': G_loss, 'D_loss': D_loss})
      if self.mixed_precision:
        G_loss = self.G_optimizer.get_scaled_loss(G_loss)
        D_loss = self.D_optimizer.get_scaled_loss(D_loss)
    self.G_optimizer.minimize(self.G, G_loss, tape)
    self.D_optimizer.minimize(self.D, D_loss, tape)
    return result

  @tf.function
  def validate(self, inputs):
    result = {}
    noise = self.sample_noise(batch_size=inputs.shape[0])
    fake = self.G(noise, training=True)
    discriminate_real = self.D(inputs, training=True)
    discriminate_fake = self.D(fake, training=True)
    G_loss = self.generator_loss(discriminate_fake)
    D_loss = self.discriminator_loss(discriminate_real, discriminate_fake)
    result.update({'G_loss': G_loss, 'D_loss': D_loss})
    return result

  @tf.function
  def generate(self, noise, denorm=False):
    fake = self.G(noise, training=False)
    if denorm:
      fake = denormalize(fake, x_min=self._signals_min, x_max=self._signals_max)
    return fake
