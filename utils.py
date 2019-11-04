import os
import io
import json
import pickle
import numpy as np
from math import ceil
import tensorflow as tf
from oasis.functions import deconvolve

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
  hparams.normalize = info['normalize']


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


def store_hparams(hparams):
  with open(os.path.join(hparams.output_dir, 'hparams.json'), 'w') as file:
    json.dump(hparams.__dict__, file)


def oasis_deconvolve(signals):
  if tf.is_tensor(signals):
    signals = signals.numpy()

  spikes = []
  for i in range(len(signals)):
    c, s, b, g, lam = deconvolve(signals[i], g=(None, None), penalty=1)
    spikes.append(s / s.max())

  return np.array(spikes, dtype=np.float32)


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
    plt.savefig(buf, dpi=100, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    return image

  def _simple_axis(self, axis):
    """plot only x and y axis, not a frame for subplot ax"""
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.get_xaxis().tick_bottom()
    axis.get_yaxis().tick_left()

  def _plot_trace(self, signal, spike):
    figure = plt.figure()
    # plot calcium signal
    plt.figure(figsize=(20, 4))
    plt.subplot(211)
    plt.plot(signal, label='signal', zorder=-12, c='r')
    plt.legend(ncol=3, frameon=False, loc=(.02, .85))
    self._simple_axis(plt.gca())
    plt.tight_layout()
    # plot spike train
    plt.subplot(212)
    plt.bar(np.arange(len(spike)), spike, width=0.4, label='oasis', color='y')
    plt.ylim(0, 1.3)
    plt.legend(ncol=3, frameon=False, loc=(.02, .85))
    self._simple_axis(plt.gca())
    plt.tight_layout()
    return self._plot_to_image(figure)

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

  def plot_traces(self, tag, signals, spikes=None, step=None, training=True):
    images = []

    if tf.is_tensor(signals):
      signals = signals.numpy()

    if spikes is None:
      spikes = oasis_deconvolve(signals)

    if tf.is_tensor(spikes):
      spikes = spikes.numpy()

    for i in range(signals.shape[0]):
      image = self._plot_trace(signals[i], spikes[i])
      images.append(image)

    images = tf.stack(images)
    self.image(tag, values=images, step=step, training=training)

  def graph(self):
    writer = self._get_writer(training=True)
    with writer.as_default():
      tf.summary.trace_export(name='models', step=0)
