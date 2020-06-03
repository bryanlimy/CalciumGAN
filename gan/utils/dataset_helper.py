import os
import pickle
import numpy as np
from math import ceil
from tqdm import tqdm
import tensorflow as tf

from . import utils
from . import h5_helper


def cache_validation_set(hparams, validation_ds):
  ''' Cache validation set as pickles for faster spike metrics evaluation '''
  if os.path.exists(hparams.validation_cache):
    return

  with tf.device('/CPU:0'):
    for signal, spike in tqdm(
        validation_ds,
        desc='Cache validation set',
        disable=not bool(hparams.verbose)):

      signal, spike = signal.numpy(), spike.numpy()

      if hparams.normalize:
        signal = utils.denormalize(
            signal, x_min=hparams.signals_min, x_max=hparams.signals_max)

      if hparams.fft:
        signal = utils.ifft(signal)

      h5_helper.write(hparams.validation_cache, {
          'signals': signal.astype(np.float32),
          'spikes': spike.astype(np.int8)
      })


def plot_real_signals(hparams, summary, ds, indexes=None):
  # plot signals and spikes from validation set
  signals, spikes = next(iter(ds))

  signals, spikes = signals[0].numpy(), spikes[0].numpy()

  if hparams.normalize:
    signals = utils.denormalize(
        signals, x_min=hparams.signals_min, x_max=hparams.signals_max)

  if hparams.fft:
    signals = utils.ifft(signals)

  signals = utils.set_array_format(signals, data_format='CW', hparams=hparams)
  spikes = utils.set_array_format(spikes, data_format='CW', hparams=hparams)

  summary.plot_traces(
      'real',
      signals,
      spikes,
      indexes=indexes if indexes is not None else hparams.focus_neurons,
      step=0,
      training=False)


def get_fashion_mnist(hparams):
  (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

  def preprocess(images):
    images = np.reshape(images, newshape=(images.shape[0], 28, 28, 1))
    return images.astype('float32') / 255.0

  x_train = preprocess(x_train)
  x_test = preprocess(x_test)

  hparams.train_size = len(x_train)
  hparams.eval_size = len(x_test)

  train_ds = tf.data.Dataset.from_tensor_slices(x_train)
  train_ds = train_ds.shuffle(buffer_size=2048)
  train_ds = train_ds.batch(hparams.batch_size)
  train_ds = train_ds.prefetch(2)

  eval_ds = tf.data.Dataset.from_tensor_slices(x_test)
  eval_ds = eval_ds.batch(hparams.batch_size)

  return train_ds, eval_ds


def get_surrogate_dataset(hparams):
  filename = os.path.join(hparams.input_dir, 'training.pkl')
  if not os.path.exists(filename):
    print('training dataset {} not found'.format(filename))
    exit()

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


def get_dataset_info(hparams):
  """ Get dataset information """
  with open(os.path.join(hparams.input_dir, 'info.pkl'), 'rb') as file:
    info = pickle.load(file)
  hparams.train_files = os.path.join(hparams.input_dir, 'train-*.record')
  hparams.validation_files = os.path.join(hparams.input_dir,
                                          'validation-*.record')
  hparams.train_size = info['train_size']
  hparams.validation_size = info['validation_size']
  hparams.signal_shape = info['signal_shape']
  hparams.spike_shape = info['spike_shape']
  hparams.sequence_length = info['sequence_length']
  hparams.num_neurons = info['num_neurons']
  hparams.num_channels = info['num_channels']
  hparams.num_train_shards = info['num_train_shards']
  hparams.num_validation_shards = info['num_validation_shards']
  hparams.buffer_size = info['buffer_size']
  hparams.normalize = info['normalize']
  hparams.fft = info['fft']

  if hparams.normalize:
    hparams.signals_min = float(info['signals_min'])
    hparams.signals_max = float(info['signals_max'])

  if hparams.save_generated:
    hparams.generated_dir = os.path.join(hparams.output_dir, 'generated')
    if not os.path.exists(hparams.generated_dir):
      os.makedirs(hparams.generated_dir)

    hparams.validation_cache = os.path.join(hparams.generated_dir,
                                            'validation.h5')


def get_tfrecords(hparams):
  if not os.path.exists(hparams.input_dir):
    print('input directory {} cannot be found'.format(hparams.input_dir))
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

  train_files = tf.data.Dataset.list_files(hparams.train_files)
  train_ds = train_files.interleave(
      tf.data.TFRecordDataset, num_parallel_calls=1)
  train_ds = train_ds.map(_parse_example, num_parallel_calls=2)
  train_ds = train_ds.cache()
  train_ds = train_ds.shuffle(hparams.buffer_size)
  train_ds = train_ds.batch(hparams.batch_size)
  train_ds = train_ds.prefetch(4)

  validation_files = tf.data.Dataset.list_files(hparams.validation_files)
  validation_ds = validation_files.interleave(
      tf.data.TFRecordDataset, num_parallel_calls=1)
  validation_ds = validation_ds.map(_parse_example, num_parallel_calls=2)
  validation_ds = validation_ds.batch(hparams.batch_size)

  return train_ds, validation_ds


def get_dataset(hparams, summary):
  hparams.noise_shape = (hparams.noise_dim,)

  if hparams.input_dir == 'fashion_mnist':
    train_ds, validation_ds = get_fashion_mnist(hparams)
  elif hparams.surrogate_ds:
    train_ds, validation_ds = get_surrogate_dataset(hparams)
    plot_real_signals(
        hparams,
        summary,
        validation_ds,
        indexes=list(range(hparams.num_neurons)))
  else:
    train_ds, validation_ds = get_tfrecords(hparams)

    if hparams.save_generated:
      cache_validation_set(hparams, validation_ds)

    plot_real_signals(hparams, summary, validation_ds)

  hparams.train_steps = ceil(hparams.train_size / hparams.batch_size)
  hparams.validation_steps = ceil(hparams.validation_size / hparams.batch_size)

  return train_ds, validation_ds
