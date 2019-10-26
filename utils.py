import os
import io
import pickle
import numpy as np
from math import ceil
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def get_fashion_mnist(hparams, summary):
  (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

  def preprocess(images):
    images = np.reshape(images, newshape=(images.shape[0], 28, 28, 1))
    return images.astype('float32') / 255.0

  x_train = preprocess(x_train)
  x_test = preprocess(x_test)

  hparams.train_size = len(x_train)
  hparams.eval_size = len(x_test)

  summary.image('real', x_test[:5], training=False)

  train_ds = tf.data.Dataset.from_tensor_slices(x_train)
  train_ds = train_ds.shuffle(buffer_size=2048)
  train_ds = train_ds.batch(hparams.batch_size)
  train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

  eval_ds = tf.data.Dataset.from_tensor_slices(x_test)
  eval_ds = eval_ds.batch(hparams.batch_size)

  return train_ds, eval_ds


def get_dataset_info(hparams):
  """ Get dataset information """
  with open(os.path.join(hparams.input, 'info.pkl'), 'rb') as file:
    info = pickle.load(file)
  hparams.train_size = info['train_size']
  hparams.eval_size = info['eval_size']
  hparams.signal_shape = info['signal_shape']
  hparams.spike_shape = info['spike_shape']
  hparams.train_shards = info['train_shards']
  hparams.eval_shards = info['eval_shards']
  hparams.buffer_size = info['num_per_shard']


def get_calcium_signals(hparams, summary):
  if not os.path.exists(hparams.input):
    print('input directory {} cannot be found'.format(hparams.input))
    exit()

  get_dataset_info(hparams)

  features_description = {
      'signal': tf.io.FixedLenFeature([], tf.string),
      'spike': tf.io.FixedLenFeature([], tf.string)
  }

  def _parse_example(example):
    parsed = tf.io.parse_single_example(example, features_description)
    signal = tf.io.decode_raw(parsed['signal'], out_type=tf.float32)
    signal = tf.reshape(signal, shape=hparams.signal_shape)
    spike = tf.io.decode_raw(parsed['spike'], out_type=tf.float32)
    spike = tf.reshape(spike, shape=hparams.spike_shape)
    return signal, spike

  train_files = tf.data.Dataset.list_files(
      os.path.join(hparams.input, 'train-*.record'))
  train_ds = train_files.interleave(tf.data.TFRecordDataset, cycle_length=4)
  train_ds = train_ds.map(
      _parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_ds = train_ds.shuffle(buffer_size=hparams.buffer_size)
  train_ds = train_ds.batch(hparams.batch_size)
  train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

  eval_files = tf.data.Dataset.list_files(
      os.path.join(hparams.input, 'eval-*.record'))
  eval_ds = eval_files.interleave(tf.data.TFRecordDataset, cycle_length=4)
  eval_ds = eval_ds.map(
      _parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  eval_ds = eval_ds.batch(hparams.batch_size)

  return train_ds, eval_ds


def get_dataset(hparams, summary):
  if hparams.input == 'fashion_mnist':
    train_ds, eval_ds = get_fashion_mnist(hparams, summary)
  else:
    train_ds, eval_ds = get_calcium_signals(hparams, summary)

  hparams.generator_input_shape = (hparams.noise_dim,)
  hparams.steps_per_epoch = ceil(hparams.train_size / hparams.batch_size)

  return train_ds, eval_ds


def derivative_mse(set1, set2):
  diff1 = np.diff(set1, n=1, axis=-1)
  diff2 = np.diff(set2, n=1, axis=-1)
  mse = np.mean(np.square(diff1 - diff2))
  return mse


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
