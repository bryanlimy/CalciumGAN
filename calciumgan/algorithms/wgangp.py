from .registry import register

import tensorflow as tf

from calciumgan.algorithms.gan import GAN
from calciumgan.utils.utils import update_dict


@register('wgangp')
class WGANGP(GAN):

  def __init__(self, hparams, G, D):
    super(WGANGP, self).__init__(hparams, G, D)

    self.penalty_coefficient = hparams.gradient_penalty
    self.n_critic = hparams.n_critic
    self.conv2d = hparams.conv2d

  def generator_loss(self, discriminate_fake):
    return -tf.reduce_mean(discriminate_fake)

  def _train_generator(self, inputs):
    result = {}
    noise = self.sample_noise(batch_size=inputs.shape[0])
    with tf.GradientTape(persistent=True) as tape:
      fake = self.G(noise, training=True)
      discriminate_fake = self.D(fake, training=True)
      G_loss = self.generator_loss(discriminate_fake)
      result.update({'G_loss': G_loss})
      if self.mixed_precision:
        G_loss = self.G_optimizer.get_scaled_loss(G_loss)
    self.G_optimizer.minimize(self.G, G_loss, tape)
    return result

  def interpolate(self, real, fake):
    shape = (real.shape[0], 1, 1, 1) if self.conv2d else (real.shape[0], 1, 1)
    alpha = tf.random.uniform(shape, minval=0.0, maxval=1.0)
    return (alpha * real) + ((1 - alpha) * fake)

  def gradient_penalty(self, real, fake, training=True):
    interpolated = self.interpolate(real, fake)
    with tf.GradientTape() as tape:
      tape.watch(interpolated)
      discriminate_interpolated = self.D(interpolated, training=training)
    gradient = tape.gradient(discriminate_interpolated, interpolated)
    norm = tf.norm(tf.reshape(gradient, shape=(gradient.shape[0], -1)), axis=1)
    penalty = tf.reduce_mean(tf.square(norm - 1.0))
    return self.penalty_coefficient * penalty

  def discriminator_loss(self, discriminate_real, discriminate_fake):
    real_loss = -tf.reduce_mean(discriminate_real)
    fake_loss = tf.reduce_mean(discriminate_fake)
    return real_loss + fake_loss

  def _train_discriminator(self, inputs):
    result = {}
    noise = self.sample_noise(batch_size=inputs.shape[0])
    with tf.GradientTape(persistent=True) as tape:
      fake = self.G(noise, training=True)
      discriminate_real = self.D(inputs, training=True)
      discriminate_fake = self.D(fake, training=True)
      D_loss = self.discriminator_loss(discriminate_real, discriminate_fake)
      gradient_penalty = self.gradient_penalty(inputs, fake, training=True)
      result.update({'D_loss': D_loss, 'gradient_penalty': gradient_penalty})
      D_loss += gradient_penalty
      if self.mixed_precision:
        D_loss = self.D_optimizer.get_scaled_loss(D_loss)
    self.D_optimizer.minimize(self.D, D_loss, tape)
    return result

  @tf.function
  def train(self, inputs):
    result = {}
    for i in range(self.n_critic):
      D_result = self._train_discriminator(inputs)
      update_dict(result, D_result)
    G_result = self._train_generator(inputs)
    update_dict(result, G_result)
    return {key: tf.reduce_mean(value) for key, value in result.items()}

  @tf.function
  def validate(self, inputs):
    result = {}
    noise = self.sample_noise(batch_size=inputs.shape[0])
    fake = self.G(noise, training=True)
    discriminate_real = self.D(inputs, training=False)
    discriminate_fake = self.D(fake, training=False)
    G_loss = self.generator_loss(discriminate_fake)
    D_loss = self.discriminator_loss(discriminate_real, discriminate_fake)
    gradient_penalty = self.gradient_penalty(inputs, fake, training=False)
    result.update({
        'G_loss': G_loss,
        'D_loss': D_loss,
        'gradient_penalty': gradient_penalty
    })
    return result
