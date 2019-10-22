import os
import io
import pickle
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def get_fashion_mnist(hparams, summary):
  (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

  def preprocess(images):
    images = images.reshape(images.shape[0], 28, 28, 1).astype('float32')
    return images / 255.0

  x_train = preprocess(x_train)
  x_test = preprocess(x_test)

  summary.image('real', x_test[:5], training=False)

  return x_train, x_test


def get_calcium_signals(hparams, summary):
  if not os.path.exists(hparams.input):
    print('input pickle {} cannot be found'.format(hparams.input))
    exit()

  with open(hparams.input, 'rb') as file:
    segments = pickle.load(file)

  np.random.shuffle(segments)

  train_size = int(len(segments) * 0.7)

  x_train = segments[:train_size]
  x_test = segments[train_size:]

  summary.plot('real', x_test[:5], training=False)

  return x_train, x_test


def get_dataset(hparams, summary):
  if hparams.input == 'fashion_mnist':
    x_train, x_test = get_fashion_mnist(hparams, summary)
  else:
    x_train, x_test = get_calcium_signals(hparams, summary)

  hparams.generator_input_shape = (hparams.noise_dim,)
  hparams.generator_output_shape = x_train.shape[1:]

  hparams.steps_per_epoch = int(np.ceil(len(x_train) / hparams.batch_size))

  # train set
  train_ds = tf.data.Dataset.from_tensor_slices(x_train)
  train_ds = train_ds.shuffle(buffer_size=2048)
  train_ds = train_ds.batch(hparams.batch_size)
  train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

  # validation set
  validation_ds = tf.data.Dataset.from_tensor_slices(x_test)
  validation_ds = validation_ds.batch(hparams.batch_size)

  return train_ds, validation_ds


class Summary(object):
  """ 
  Log tf.Summary to output_dir during training and output_dir/eval during 
  evaluation
  """

  def __init__(self, hparams):
    self._hparams = hparams
    self.train_writer = tf.summary.create_file_writer(hparams.output_dir)
    self.val_writer = tf.summary.create_file_writer(
        os.path.join(hparams.output_dir, 'validation'))
    tf.summary.trace_on(graph=True, profiler=False)

  def _get_writer(self, training):
    return self.train_writer if training else self.val_writer

  def _get_step(self):
    return self._hparams.global_step

  def _plot_to_image(self, figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    return image

  def scalar(self, tag, value, step=None, training=True):
    writer = self._get_writer(training)
    step = self._get_step() if step is None else step
    with writer.as_default():
      tf.summary.scalar(tag, value, step=step)

  def histogram(self, tag, values, step=None, training=True):
    writer = self._get_writer(training)
    step = self._get_step() if step is None else step
    with writer.as_default():
      tf.summary.histogram(tag, values, step=step)

  def image(self, tag, values, step=None, training=True):
    writer = self._get_writer(training)
    step = self._get_step() if step is None else step
    with writer.as_default():
      tf.summary.image(tag, data=values, step=step, max_outputs=values.shape[0])

  def plot(self, tag, values, step=None, training=True):
    images = []
    for i in range(values.shape[0]):
      value = values[i]
      figure = plt.figure()
      plt.plot(value)
      plt.xlabel('Time (ms)')
      plt.ylabel('Activity')
      image = self._plot_to_image(figure)
      images.append(image)
    images = tf.stack(images)
    self.image(tag, values=images, step=step, training=training)

  def graph(self):
    writer = self._get_writer(training=True)
    with writer.as_default():
      tf.summary.trace_export(name='models', step=0)
