import os
import pickle
import numpy as np
from math import ceil
import tensorflow as tf

from calciumgan.utils import utils


def get_surrogate_dataset(hparams):
  filename = os.path.join(hparams.input_dir, 'training.pkl')
  if not os.path.exists(filename):
    raise FileNotFoundError('training dataset {} not found'.format(filename))

  with open(filename, 'rb') as file:
    data = pickle.load(file)

  def normalize(x):
    hparams.signals_min = float(np.min(x))
    hparams.signals_max = float(np.max(x))
    shape = x.shape
    x = np.reshape(x, newshape=(shape[0], shape[1] * shape[2]))
    x = (x - hparams.signals_min) / (hparams.signals_max - hparams.signals_min)
    x = np.reshape(x, newshape=shape)
    return x

  # set shape to (num trials, sequence length, num neurons)
  signals = np.transpose(data['signals'], axes=[0, 2, 1])

  signals, spikes = normalize(signals), data['spikes']
  train_size = 8192
  train_signals, train_spikes = signals[:train_size], spikes[:train_size]
  test_signals, test_spikes = signals[train_size:], spikes[train_size:]

  hparams.train_size = len(train_signals)
  hparams.validation_size = len(test_signals)
  hparams.signal_shape = train_signals.shape[1:]
  hparams.spike_shape = data['spikes'].shape[1:]
  hparams.sequence_length = train_signals.shape[1]
  hparams.num_neurons = train_signals.shape[-1]
  hparams.num_channels = train_signals.shape[-1]
  hparams.normalize = True
  hparams.fft = False
  hparams.conv2d = False

  if hparams.save_generated:
    hparams.generated_dir = os.path.join(hparams.output_dir, 'generated')
    if not os.path.exists(hparams.generated_dir):
      os.makedirs(hparams.generated_dir)

    hparams.validation_cache = os.path.join(hparams.generated_dir,
                                            'validation.h5')

  train_ds = tf.data.Dataset.from_tensor_slices((train_signals, train_spikes))
  train_ds = train_ds.shuffle(buffer_size=2048)
  train_ds = train_ds.batch(hparams.batch_size)
  train_ds = train_ds.prefetch(4)

  validation_ds = tf.data.Dataset.from_tensor_slices((test_signals,
                                                      test_spikes))
  validation_ds = validation_ds.batch(hparams.batch_size)

  return train_ds, validation_ds


def get_tfrecords(hparams):
  if not os.path.exists(hparams.input_dir):
    print('input directory {} cannot be found'.format(hparams.input_dir))
    exit()

  # retrieve dataset information
  info = utils.load_json(os.path.join(hparams.input_dir, 'info.json'))

  hparams.train_size = info['train_size']
  hparams.validation_size = info['validation_size']
  hparams.signal_shape = info['signal_shape']
  hparams.sequence_length = info['sequence_length']
  hparams.num_neurons = info['num_neurons']
  hparams.num_channels = info['num_channels']
  hparams.num_train_shards = info['num_train_shards']
  hparams.num_validation_shards = info['num_validation_shards']
  hparams.buffer_size = info['buffer_size']
  hparams.normalize = info['normalize']
  hparams.fft = info['fft']
  hparams.conv2d = info['conv2d']
  hparams.signals_min = float(info['signals_min'])
  hparams.signals_max = float(info['signals_max'])

  features_description = {'signal': tf.io.FixedLenFeature([], tf.string)}

  def _parse_example(example):
    parsed = tf.io.parse_single_example(example, features_description)
    signal = tf.io.decode_raw(parsed['signal'], out_type=tf.float32)
    signal = tf.reshape(signal, shape=hparams.signal_shape)
    return signal

  train_files = tf.data.Dataset.list_files(
      os.path.join(hparams.input_dir, 'train-*.record'))
  train_ds = train_files.interleave(
      tf.data.TFRecordDataset, num_parallel_calls=1)
  train_ds = train_ds.map(_parse_example, num_parallel_calls=2)
  train_ds = train_ds.cache()
  train_ds = train_ds.shuffle(hparams.buffer_size)
  train_ds = train_ds.batch(hparams.batch_size)
  train_ds = train_ds.prefetch(4)

  validation_files = tf.data.Dataset.list_files(
      os.path.join(hparams.input_dir, 'validation-*.record'))
  validation_ds = validation_files.interleave(
      tf.data.TFRecordDataset, num_parallel_calls=1)
  validation_ds = validation_ds.map(_parse_example, num_parallel_calls=2)
  validation_ds = validation_ds.batch(hparams.batch_size)

  return train_ds, validation_ds


def get_dataset(hparams, summary):
  hparams.noise_shape = (hparams.noise_dim,)

  if hparams.surrogate_ds:
    train_ds, validation_ds = get_surrogate_dataset(hparams)
  else:
    train_ds, validation_ds = get_tfrecords(hparams)

  utils.plot_samples(
      hparams, summary, next(iter(validation_ds)), step=0, tag='real_traces')

  hparams.train_steps = ceil(hparams.train_size / hparams.batch_size)
  hparams.validation_steps = ceil(hparams.validation_size / hparams.batch_size)
  hparams.samples_dir = os.path.join(hparams.output_dir, 'samples')
  hparams.checkpoint_dir = os.path.join(hparams.output_dir, 'checkpoint')

  return train_ds, validation_ds
