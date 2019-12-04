import os
import h5py
import json
import pickle
import numpy as np
from math import ceil
import tensorflow as tf

from .oasis_helper import deconvolve_signals


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
  hparams.mean_spike_count = float(info['mean_spike_count'])


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


def get_signal_filename(hparams, epoch):
  """ return the filename of the signal h5 file given epoch """
  return os.path.join(hparams.output_dir,
                      'epoch{:03d}_signals.h5'.format(epoch))


def append_h5(ds, value):
  """ append value to a H5 dataset """
  if type(value) != np.ndarray:
    value = np.array(value, dtype=np.float32)
  ds.resize((ds.shape[0] + value.shape[0]), axis=0)
  ds[-value.shape[0]:] = value


def create_or_append_h5(file, name, value):
  """ create or append value to a H5 dataset """
  if name in file:
    append_h5(file[name], value)
  else:
    file.create_dataset(
        name,
        dtype=np.float32,
        data=value,
        chunks=True,
        maxshape=(None, value.shape[1]))


def save_signals(hparams, epoch, real_spikes, real_signals, fake_signals):
  filename = get_signal_filename(hparams, epoch)

  with h5py.File(filename, 'a') as file:
    create_or_append_h5(file, 'real_spikes', real_spikes)
    create_or_append_h5(file, 'real_signals', real_signals)
    create_or_append_h5(file, 'fake_signals', fake_signals)


def deconvolve_saved_signals(hparams, epoch):
  filename = get_signal_filename(hparams, epoch)

  with h5py.File(filename, 'a') as file:
    fake_signals = file['fake_signals'][:]

    fake_spikes = deconvolve_signals(fake_signals, multiprocessing=True)
    file.create_dataset(
        'fake_spikes',
        dtype=np.float32,
        data=fake_spikes,
        chunks=True,
        maxshape=(None, fake_spikes.shape[1]))

  return fake_spikes
