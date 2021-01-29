import tensorflow as tf
from tensorflow.keras import mixed_precision


class Optimizer(object):

  def __init__(self, hparams):
    self.mixed_precision = hparams.mixed_precision
    self.optimizer = tf.keras.optimizers.Adam(hparams.learning_rate)
    if self.mixed_precision:
      self.optimizer = mixed_precision.LossScaleOptimizer(
          self.optimizer, dynamic=True)

  @property
  def iterations(self):
    return self.optimizer.iterations

  @iterations.setter
  def iterations(self, value):
    self.optimizer.iterations = value

  def get_scaled_loss(self, loss):
    return self.optimizer.get_scaled_loss(loss)

  def get_unscaled_gradients(self, scaled_gradients):
    return self.optimizer.get_unscaled_gradients(scaled_gradients)

  def minimize(self, model, loss, tape):
    gradients = tape.gradient(loss, model.trainable_variables)
    if self.mixed_precision:
      gradients = self.get_unscaled_gradients(gradients)
    self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
