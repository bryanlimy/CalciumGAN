from .registry import register

import tensorflow as tf


@register('gan')
class GAN(object):

  def __init__(self, hparams, generator, discriminator, summary):
    self.generator = generator
    self.discriminator = discriminator

    self._summary = summary
    self._num_neurons = hparams.num_neurons
    self._noise_dim = hparams.noise_dim
    self._signals_min = hparams.signals_min
    self._signals_max = hparams.signals_max
    self._normalize = hparams.normalize

    self.gen_optimizer = tf.keras.optimizers.Adam(hparams.learning_rate)
    self.dis_optimizer = tf.keras.optimizers.Adam(hparams.learning_rate)

    self._cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  def denormalize(self, x):
    ''' re-scale signals back to its original range '''
    return x * (self._signals_max - self._signals_min) + self._signals_min

  def mean_signals_error(self, real, fake):
    ''' return the MSE of the mean of the real and fake traces '''
    if self._normalize:
      real = self.denormalize(real)
      fake = self.denormalize(fake)
    return tf.reduce_mean(
        tf.square(tf.reduce_mean(real) - tf.reduce_mean(fake)))

  def kl_divergence(self, real, fake):
    return tf.reduce_mean(tf.keras.losses.KLD(y_true=real, y_pred=fake))

  def metrics(self, real, fake):
    return {
        'kl_divergence': self.kl_divergence(real, fake),
        'mean_signals_error': self.mean_signals_error(real, fake)
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
    noise = tf.random.normal((inputs.shape[0], self._noise_dim))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
      _, gen_loss, dis_loss, gradient_penalty, metrics = self._step(
          inputs, noise)

    gen_gradients = gen_tape.gradient(gen_loss,
                                      self.generator.trainable_variables)
    dis_gradients = dis_tape.gradient(dis_loss,
                                      self.discriminator.trainable_variables)

    self.gen_optimizer.apply_gradients(
        zip(gen_gradients, self.generator.trainable_variables))
    self.dis_optimizer.apply_gradients(
        zip(dis_gradients, self.discriminator.trainable_variables))

    return gen_loss, dis_loss, gradient_penalty, metrics

  @tf.function
  def validate(self, inputs):
    noise = tf.random.normal((inputs.shape[0], self._noise_dim))
    return self._step(inputs, noise, training=False)

  @tf.function
  def samples(self, noise):
    return self.generator(noise, training=False)
