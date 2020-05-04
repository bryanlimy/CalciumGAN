import os

# use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pickle
import argparse
import numpy as np
from math import ceil
import tensorflow as tf
from shutil import rmtree

np.random.seed(1234)


def split(sequence, n):
  """ divide sequence into n sub-sequence evenly """
  k, m = divmod(len(sequence), n)
  return [
      sequence[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)
  ]


def normalize(x, x_min, x_max):
  """ scale x to be between 0 and 1 """
  return (x - x_min) / (x_max - x_min)


def fft(x):
  """ Apply FFT on x and represent complex number in a 2D float array"""
  x = np.fft.fft(x.astype(np.complex64), axis=-1, norm='ortho')

  return np.concatenate([np.real(x), np.imag(x)], axis=-1)


def calculate_num_per_shard(hparams):
  """ 
  calculate the number of data per shard given sequence_length such that each 
  shard is target_size GB
  """
  num_per_shard = ceil((120 / hparams.sequence_length) * 1100) * 10  # 1GB shard
  if hparams.fft:
    num_per_shard *= 2 / 3
  return int(num_per_shard * hparams.target_shard_size)


def get_segments(hparams):
  print('processing file {}...'.format(hparams.input))

  assert hparams.stride >= 1

  with open(hparams.input, 'rb') as file:
    data = pickle.load(file)

  raw_signals = np.array(data['signals'], dtype=np.float32)
  raw_spikes = np.array(data['oasis'], dtype=np.float32)

  # remove first two rows in signals
  raw_signals = raw_signals[2:]
  raw_spikes = raw_spikes[2:]

  assert raw_signals.shape == raw_spikes.shape

  # max and min value of signals
  hparams.signals_min = np.min(raw_signals)
  hparams.signals_max = np.max(raw_signals)

  print('signals min {:.04f}, max {:.04f}, mean {:.04f}'.format(
      np.min(raw_signals), np.max(raw_signals), np.mean(raw_signals)))

  if hparams.normalize:
    print('apply normalization')
    raw_signals = normalize(raw_signals, hparams.signals_min,
                            hparams.signals_max)
    print('signals min {:.04f}, max {:.04f}, mean {:.04f}'.format(
        np.min(raw_signals), np.max(raw_signals), np.mean(raw_signals)))

  # set signals and spikes to WC [sequence, num. neurons]
  raw_signals = np.swapaxes(raw_signals, 0, 1)
  raw_spikes = np.swapaxes(raw_spikes, 0, 1)

  hparams.num_neurons = raw_signals.shape[1]
  hparams.num_channels = hparams.num_neurons

  if hparams.fft:
    print('apply fft')
    raw_signals = fft(raw_signals)
    print('signals min {:.04f}, max {:.04f}, mean {:.04f}'.format(
        np.min(raw_signals), np.max(raw_signals), np.mean(raw_signals)))
    hparams.num_channels = raw_signals.shape[-1]

  print('segmentation with stride {}'.format(hparams.stride))

  signals, spikes = [], []

  i = 0
  while i + hparams.sequence_length < raw_signals.shape[0]:
    signals.append(raw_signals[i:i + hparams.sequence_length, :])
    spikes.append(raw_spikes[i:i + hparams.sequence_length, :])
    i += hparams.stride

  signals = np.array(signals, dtype=np.float32)
  spikes = np.array(spikes, dtype=np.float32)

  return signals, spikes


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(signal, spike):
  features = {
      'signal': _bytes_feature(signal.tostring()),
      'spike': _bytes_feature(spike.tostring())
  }
  example = tf.train.Example(features=tf.train.Features(feature=features))
  return example.SerializeToString()


def get_record_filename(hparams, mode, shard_id, num_shards):
  filename = '{}-{:03d}-of-{:03d}.record'.format(mode, shard_id + 1, num_shards)
  return os.path.join(hparams.output_dir, filename)


def write_to_record(hparams, mode, shard, num_shards, signals, spikes, indexes):
  record_filename = get_record_filename(hparams, mode, shard, num_shards)
  print('writing {} segments to {}...'.format(len(signals), record_filename))

  with tf.io.TFRecordWriter(record_filename) as writer:
    for i in indexes:
      example = serialize_example(signals[i], spikes[i])
      writer.write(example)


def write_to_records(hparams, mode, signals, spikes, indexes):
  if not os.path.exists(hparams.output_dir):
    os.makedirs(hparams.output_dir)

  # calculate the number of records to create
  num_shards = 1 if hparams.num_per_shard == 0 else ceil(
      len(indexes) / hparams.num_per_shard)

  print('writing {} segments to {} {} records...'.format(
      len(indexes), num_shards, mode))

  if mode == 'train':
    hparams.num_train_shards = num_shards
  else:
    hparams.num_validation_shards = num_shards

  sharded_indexes = split(indexes, num_shards)

  for shard in range(num_shards):
    write_to_record(
        hparams,
        mode=mode,
        shard=shard,
        num_shards=num_shards,
        signals=signals,
        spikes=spikes,
        indexes=sharded_indexes[shard],
    )


def main(hparams):
  if not os.path.exists(hparams.input):
    print('input file {} does not exists'.format(hparams.input))
    exit()

  if os.path.exists(hparams.output_dir):
    if hparams.replace:
      rmtree(hparams.output_dir)
    else:
      print('output directory {} already exists\n'.format(hparams.output_dir))
      exit()

  signals, spikes = get_segments(hparams)

  # shuffle data
  indexes = np.arange(len(signals))
  np.random.shuffle(indexes)

  hparams.train_size = int(len(signals) * hparams.train_percentage)
  hparams.validation_size = len(signals) - hparams.train_size
  hparams.signal_shape = signals.shape[1:]
  hparams.spike_shape = spikes.shape[1:]

  hparams.num_per_shard = calculate_num_per_shard(hparams)

  print('{} segments in each shard with target shard size {}'.format(
      hparams.num_per_shard, hparams.target_shard_size))

  write_to_records(
      hparams,
      mode='train',
      signals=signals,
      spikes=spikes,
      indexes=indexes[:hparams.train_size])

  write_to_records(
      hparams,
      mode='validation',
      signals=signals,
      spikes=spikes,
      indexes=indexes[hparams.train_size:])

  # save information of the dataset
  with open(os.path.join(hparams.output_dir, 'info.pkl'), 'wb') as file:
    info = {
        'train_size': hparams.train_size,
        'validation_size': hparams.validation_size,
        'signal_shape': hparams.signal_shape,
        'spike_shape': hparams.spike_shape,
        'sequence_length': hparams.sequence_length,
        'num_neurons': hparams.num_neurons,
        'num_channels': hparams.num_channels,
        'num_train_shards': hparams.num_train_shards,
        'num_validation_shards': hparams.num_validation_shards,
        'buffer_size': min(hparams.num_per_shard, hparams.train_size),
        'normalize': hparams.normalize,
        'stride': hparams.stride,
        'fft': hparams.fft
    }
    if hparams.normalize:
      info['signals_min'] = hparams.signals_min
      info['signals_max'] = hparams.signals_max
    pickle.dump(info, file)

  print('saved {} TFRecords to {}'.format(
      hparams.num_train_shards + hparams.num_validation_shards,
      hparams.output_dir))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input', default='raw_data/ST260_Day4_signals4Bryan.pkl', type=str)
  parser.add_argument('--output_dir', default='tfrecords', type=str)
  parser.add_argument('--sequence_length', default=2048, type=int)
  parser.add_argument('--stride', default=2, type=int)
  parser.add_argument('--normalize', action='store_true')
  parser.add_argument('--fft', action='store_true')
  parser.add_argument('--replace', action='store_true')
  parser.add_argument('--train_percentage', default=0.9, type=float)
  parser.add_argument(
      '--target_shard_size',
      default=0.25,
      type=float,
      help='target size in GB for each TFRecord file.')
  hparams = parser.parse_args()

  main(hparams)
