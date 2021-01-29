import os

# use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pickle
import argparse
import numpy as np
from math import ceil
from tqdm import tqdm
import tensorflow as tf
from shutil import rmtree

from calciumgan.utils import utils

np.random.seed(1234)


def get_segments(hparams):
  print('processing file {}...'.format(hparams.input))

  assert hparams.stride >= 1

  with open(hparams.input, 'rb') as file:
    data = pickle.load(file)

  raw_signals = np.array(data['signals'], dtype=np.float32)

  # remove first two rows in signals
  if not hparams.is_dg_data:
    raw_signals = raw_signals[2:]

  # set signals and spikes to WC [sequence, num. neurons, ...]
  raw_signals = np.swapaxes(raw_signals, 0, 1)

  hparams.num_neurons = raw_signals.shape[1]
  hparams.num_channels = hparams.num_neurons

  print(f'segmentation with stride {hparams.stride}')
  signals, i = [], 0
  while i + hparams.sequence_length < raw_signals.shape[0]:
    signals.append(raw_signals[i:i + hparams.sequence_length, ...])
    i += hparams.stride

  signals = np.array(signals, dtype=np.float32)

  if hparams.fft:
    print('\napply fft')
    signals = utils.fft(signals)
    hparams.num_channels = signals.shape[-1]

  if hparams.conv2d:
    print('\nconvert to 3D matrix')
    if hparams.fft:
      # convert matrix to [sequence, num. neurons, 2]
      mid = signals.shape[-1] // 2
      real = np.expand_dims(signals[..., :mid], axis=-1)
      imag = np.expand_dims(signals[..., mid:], axis=-1)
      signals = np.concatenate((real, imag), axis=-1)
    else:
      # convert matrix to [sequence, num. neurons, 1]
      signals = np.expand_dims(signals, axis=-1)
    hparams.num_channels = signals.shape[-1]
    print(f'signals shape {signals.shape}')

  print(f'signals min {np.min(signals):.04f}, '
        f'max {np.max(signals):.04f}, '
        f'mean {np.mean(signals):.04f}')

  # normalize signals to [0, 1]
  hparams.signals_min = np.min(signals)
  hparams.signals_max = np.max(signals)
  if hparams.normalize:
    print('\napply normalization')
    signals = utils.normalize(signals, hparams.signals_min, hparams.signals_max)
    print(f'signals min {np.min(signals):.04f}, '
          f'max {np.max(signals):.04f}, '
          f'mean {np.mean(signals):.04f}')

  print(f'\nsignals shape {signals.shape}')

  return signals


def calculate_num_per_shard(hparams):
  """ 
  calculate the number of data per shard given sequence_length such that each 
  shard is target_size GB
  """
  num_per_shard = ceil((120 / hparams.sequence_length) * 1100) * 20  # 1GB shard
  if hparams.fft:
    num_per_shard *= 2 / 3
  return int(num_per_shard * hparams.target_shard_size)


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(signal):
  features = {'signal': _bytes_feature(signal.tobytes())}
  example = tf.train.Example(features=tf.train.Features(feature=features))
  return example.SerializeToString()


def get_record_filename(hparams, prefix, shard_id, num_shards):
  filename = '{}-{:03d}-of-{:03d}.record'.format(prefix, shard_id + 1,
                                                 num_shards)
  return os.path.join(hparams.output_dir, filename)


def write_to_record(hparams, signals, shard, num_shards, indexes, prefix):
  record_filename = get_record_filename(hparams, prefix, shard, num_shards)
  print('writing {} segments to {}...'.format(len(indexes), record_filename))

  with tf.io.TFRecordWriter(record_filename) as writer:
    for i in indexes:
      example = serialize_example(signals[i])
      writer.write(example)


def write_to_records(hparams, signals, indexes, prefix, training):
  if not os.path.exists(hparams.output_dir):
    os.makedirs(hparams.output_dir)

  # calculate the number of records to create
  num_shards = 1 if hparams.num_per_shard == 0 else ceil(
      len(indexes) / hparams.num_per_shard)

  print(f'writing {len(indexes)} segments to {num_shards} records...')

  if training:
    hparams.num_train_shards = num_shards
  else:
    hparams.num_validation_shards = num_shards

  sharded_indexes = utils.split(indexes, num_shards)

  for shard in range(num_shards):
    write_to_record(
        hparams,
        signals=signals,
        shard=shard,
        num_shards=num_shards,
        indexes=sharded_indexes[shard],
        prefix=prefix)


def main(hparams):
  if not os.path.exists(hparams.input):
    raise FileNotFoundError(f'input file {hparams.input} does not exists')

  if os.path.exists(hparams.output_dir):
    if hparams.overwrite:
      rmtree(hparams.output_dir)
    else:
      raise FileExistsError(
          f'output directory {hparams.output_dir} already exists')

  signals = get_segments(hparams)

  # shuffle data
  indexes = np.arange(len(signals))
  np.random.shuffle(indexes)

  hparams.train_size = len(signals) - hparams.validation_size
  hparams.signal_shape = signals.shape[1:]

  hparams.num_per_shard = calculate_num_per_shard(hparams)

  print('\n{} segments in each shard with target shard size {}'.format(
      hparams.num_per_shard, hparams.target_shard_size))

  hparams.train_prefix = 'train'
  write_to_records(
      hparams,
      signals=signals,
      indexes=indexes[:hparams.train_size],
      prefix=hparams.train_prefix,
      training=True)

  hparams.validation_prefix = 'validation'
  write_to_records(
      hparams,
      signals=signals,
      indexes=indexes[hparams.train_size:],
      prefix=hparams.validation_prefix,
      training=False)

  # save information of the dataset
  utils.save_json(
      os.path.join(hparams.output_dir, 'info.json'),
      data={
          'train_size': hparams.train_size,
          'validation_size': hparams.validation_size,
          'signal_shape': hparams.signal_shape,
          'sequence_length': hparams.sequence_length,
          'num_neurons': hparams.num_neurons,
          'num_channels': hparams.num_channels,
          'num_train_shards': hparams.num_train_shards,
          'num_validation_shards': hparams.num_validation_shards,
          'buffer_size': min(hparams.num_per_shard, hparams.train_size),
          'normalize': hparams.normalize,
          'fft': hparams.fft,
          'conv2d': hparams.conv2d,
          'signals_min': hparams.signals_min,
          'signals_max': hparams.signals_max,
          'train_prefix': hparams.train_prefix,
          'validation_prefix': hparams.validation_prefix,
          'is_dg_data': hparams.is_dg_data
      })

  print(f'saved TFRecords to {hparams.output_dir}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input', default='raw_data/ST260_Day4_signals4Bryan.pkl', type=str)
  parser.add_argument('--output_dir', default='tfrecords', type=str)
  parser.add_argument('--sequence_length', default=2048, type=int)
  parser.add_argument('--stride', default=2, type=int)
  parser.add_argument('--normalize', action='store_true')
  parser.add_argument('--fft', action='store_true')
  parser.add_argument('--conv2d', action='store_true')
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--validation_size', default=500, type=int)
  parser.add_argument('--is_dg_data', action='store_true')
  parser.add_argument(
      '--target_shard_size',
      default=0.5,
      type=float,
      help='target size in GB for each TFRecord file.')

  main(parser.parse_args())
