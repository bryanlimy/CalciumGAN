import os
import pickle
import numpy as np
from math import ceil
import tensorflow as tf
from multiprocessing import Pool

from . import utils
from .spike_helper import numpy_to_neo_trains

AUTOTUNE = tf.data.experimental.AUTOTUNE


def preprocess_and_cache_neuron(hparams, neuron, signals, spikes):
  filename = utils.get_real_neuron_filename(hparams, neuron)

  if hparams.normalize:
    signals = utils.denormalize(
        signals, x_min=hparams.signals_min, x_max=hparams.signals_max)

  neo_trains = numpy_to_neo_trains(spikes)
  with open(filename, 'wb') as file:
    pickle.dump({'real_signals': signals, 'real_spikes': neo_trains}, file)


def cache_validation_set(hparams, validation_ds):
  ''' Cache validation set as pickles for faster spike metrics evaluation '''
  if hparams.verbose:
    print('Cache validation dataset to {}'.format(hparams.validation_dir))

  if not os.path.exists(hparams.validation_dir):
    os.makedirs(hparams.validation_dir)

  real_signals, real_spikes = [], []
  for signals, spikes in validation_ds:
    real_signals.append(signals.numpy())
    real_spikes.append(spikes.numpy())

  real_signals = np.concatenate(real_signals, axis=0)
  real_signals = utils.swap_neuron_major(hparams, real_signals)

  real_spikes = np.concatenate(real_spikes, axis=0)
  real_spikes = utils.swap_neuron_major(hparams, real_spikes)

  pool = Pool(processes=hparams.num_processors)
  pool.starmap(preprocess_and_cache_neuron, [(
      hparams,
      neuron,
      real_signals[neuron],
      real_spikes[neuron],
  ) for neuron in range(len(real_spikes))])
  pool.close()


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
  train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

  eval_ds = tf.data.Dataset.from_tensor_slices(x_test)
  eval_ds = eval_ds.batch(hparams.batch_size)

  return train_ds, eval_ds


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
  hparams.num_train_shards = info['num_train_shards']
  hparams.num_validation_shards = info['num_validation_shards']
  hparams.buffer_size = info['buffer_size']
  hparams.normalize = info['normalize']

  hparams.signals_min = float(info['signals_min'])
  hparams.signals_max = float(info['signals_max'])

  if hparams.spike_metrics:
    hparams.generated_dir = os.path.join(hparams.output_dir, 'generated')
    if not os.path.exists(hparams.generated_dir):
      os.makedirs(hparams.generated_dir)

    # directory to store preprocessed validation data
    hparams.validation_dir = os.path.join(hparams.generated_dir, 'validation')


def get_calcium_signals(hparams):
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
      tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE)
  train_ds = train_ds.map(_parse_example, num_parallel_calls=AUTOTUNE)
  train_ds = train_ds.shuffle(hparams.buffer_size)
  train_ds = train_ds.batch(hparams.batch_size)
  train_ds = train_ds.prefetch(AUTOTUNE)

  validation_files = tf.data.Dataset.list_files(hparams.validation_files)
  validation_ds = validation_files.interleave(
      tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE)
  validation_ds = validation_ds.map(_parse_example, num_parallel_calls=AUTOTUNE)
  validation_ds = validation_ds.batch(hparams.batch_size)

  return train_ds, validation_ds


def get_dataset(hparams, summary):
  hparams.noise_shape = (hparams.noise_dim,)

  if hparams.input_dir == 'fashion_mnist':
    train_ds, validation_ds = get_fashion_mnist(hparams)
  else:
    train_ds, validation_ds = get_calcium_signals(hparams)
    hparams.num_neurons = hparams.signal_shape[0]

    if hparams.spike_metrics:
      cache_validation_set(hparams, validation_ds)

    # plot signals and spikes from validation set
    sample_signals, sample_spikes = next(iter(validation_ds))
    sample_signals = utils.denormalize(
        sample_signals, x_min=hparams.signals_min, x_max=hparams.signals_max)
    summary.plot_traces(
        'real', signals=sample_signals, spikes=sample_spikes, training=False)

  hparams.train_steps = ceil(hparams.train_size / hparams.batch_size)
  hparams.validation_steps = ceil(hparams.validation_size / hparams.batch_size)

  return train_ds, validation_ds
