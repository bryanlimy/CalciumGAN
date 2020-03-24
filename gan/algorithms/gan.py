from .registry import register

import tensorflow as tf

from .optimizer import Optimizer
from ..utils import signals_metrics
from ..utils.utils import denormalize


@register('gan')
class GAN(object):

  def __init__(self, hparams, generator, discriminator, summary):
    self.generator = generator
    self.discriminator = discriminator

    self._summary = summary
    self._noise_dim = hparams.noise_dim
    self._normalize = hparams.normalize
    if hparams.normalize:
      self._signals_min = hparams.signals_min
      self._signals_max = hparams.signals_max

    self.gen_optimizer = Optimizer(hparams)
    self.dis_optimizer = Optimizer(hparams)

    self._cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  def get_noise(self, batch_size):
    return tf.random.normal((batch_size, self._noise_dim))

  def metrics(self, real, fake):
    if self._normalize:
      real = denormalize(real, x_min=self._signals_min, x_max=self._signals_max)
      fake = denormalize(fake, x_min=self._signals_min, x_max=self._signals_max)
    return {
        'signals_metrics/min': signals_metrics.min_signals_error(real, fake),
        'signals_metrics/max': signals_metrics.max_signals_error(real, fake),
        'signals_metrics/mean': signals_metrics.mean_signals_error(real, fake),
        'signals_metrics/std': signals_metrics.std_signals_error(real, fake)
    }

  def generator_loss(self, fake_output):
    return self._cross_entropy(tf.ones_like(fake_output), fake_output)

  def discriminator_loss(self,
                         real_output,
                         fake_output,
                         real=None,
                         fake=None,
                         training=True):
    real_loss = self._cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = self._cross_entropy(tf.zeros_like(fake_output), fake_output)
    gradient_penalty = None
    loss = real_loss + fake_loss
    return loss, gradient_penalty

  def _step(self, real, noise, training=True):
    fake = self.generator(noise, training=training)

    real_output = self.discriminator(real, training=training)
    fake_output = self.discriminator(fake, training=training)

    gen_loss = self.generator_loss(fake_output)
    dis_loss, gradient_penalty = self.discriminator_loss(
        real_output, fake_output, real=real, fake=fake, training=training)

    metrics = self.metrics(real=real, fake=fake)

    return fake, gen_loss, dis_loss, gradient_penalty, metrics

  @tf.function
  def train(self, inputs):
    noise = self.get_noise(batch_size=inputs.shape[0])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
      _, gen_loss, dis_loss, gradient_penalty, metrics = self._step(
          inputs, noise)
      gen_scaled_loss = self.gen_optimizer.get_scaled_loss(gen_loss)
      dis_scaled_loss = self.dis_optimizer.get_scaled_loss(dis_loss)

    self.gen_optimizer.update(self.generator, gen_scaled_loss, gen_tape)
    self.dis_optimizer.update(self.discriminator, dis_scaled_loss, dis_tape)

    return gen_loss, dis_loss, gradient_penalty, metrics

  @tf.function
  def validate(self, inputs):
    noise = self.get_noise(batch_size=inputs.shape[0])
    return self._step(inputs, noise, training=False)

  @tf.function
  def generate(self, noise, denorm=False):
    fake = self.generator(noise, training=False)
    if denorm:
      fake = denormalize(fake, x_min=self._signals_min, x_max=self._signals_max)
    return fake
