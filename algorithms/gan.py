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

    self.gen_optimizer = tf.keras.optimizers.Adam(hparams.learning_rate)
    self.dis_optimizer = tf.keras.optimizers.Adam(hparams.learning_rate)

    self._cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  def kl_divergence(self, real, fake):
    return tf.reduce_mean(tf.keras.losses.KLD(y_true=real, y_pred=fake))

  def generator_loss(self, fake_output):
    return self._cross_entropy(tf.ones_like(fake_output), fake_output)

  def discriminator_loss(self, real_output, fake_output, real=None, fake=None):
    real_loss = self._cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = self._cross_entropy(tf.zeros_like(fake_output), fake_output)
    gradient_penalty = None
    loss = real_loss + fake_loss
    return loss, gradient_penalty

  def _step(self, inputs, noise, training=True):
    fake = self.generator(noise, training=training)

    real_output = self.discriminator(inputs, training=training)
    fake_output = self.discriminator(fake, training=training)

    gen_loss = self.generator_loss(fake_output)
    dis_loss, gradient_penalty = self.discriminator_loss(
        real_output, fake_output, real=inputs, fake=fake)

    kl = self.kl_divergence(real=inputs, fake=fake)

    return fake, gen_loss, dis_loss, gradient_penalty, kl

  @tf.function
  def train(self, inputs):
    noise = tf.random.normal((inputs.shape[0], self._num_neurons,
                              self._noise_dim))
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
      _, gen_loss, dis_loss, gradient_penalty, kl = self._step(inputs, noise)

    gen_gradients = gen_tape.gradient(gen_loss,
                                      self.generator.trainable_variables)
    dis_gradients = dis_tape.gradient(dis_loss,
                                      self.discriminator.trainable_variables)

    self.gen_optimizer.apply_gradients(
        zip(gen_gradients, self.generator.trainable_variables))
    self.dis_optimizer.apply_gradients(
        zip(dis_gradients, self.discriminator.trainable_variables))

    return gen_loss, dis_loss, gradient_penalty, kl

  @tf.function
  def validate(self, inputs):
    noise = tf.random.normal((inputs.shape[0], self._num_neurons,
                              self._noise_dim))
    return self._step(inputs, noise, training=False)

  @tf.function
  def samples(self, noise):
    return self.generator(noise, training=False)
