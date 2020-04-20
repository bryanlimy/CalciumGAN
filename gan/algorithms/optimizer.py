import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision


class Optimizer(object):

  def __init__(self, hparams):
    self._mixed_precision = hparams.mixed_precision
    optimizer = tf.keras.optimizers.Adam(hparams.learning_rate)
    if self._mixed_precision:
      optimizer = mixed_precision.LossScaleOptimizer(
          optimizer, loss_scale='dynamic')
    self.optimizer = optimizer

  @property
  def iterations(self):
    return self.optimizer.iterations

  @iterations.setter
  def iterations(self, value):
    self.optimizer.iterations = value

  def get_scaled_loss(self, loss):
    return self.optimizer.get_scaled_loss(
        loss) if self._mixed_precision else loss

  def get_unscaled_gradients(self, scaled_gradients):
    return self.optimizer.get_unscaled_gradients(
        scaled_gradients) if self._mixed_precision else scaled_gradients

  def update(self, model, scaled_loss, tape):
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = self.get_unscaled_gradients(scaled_gradients)
    self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
