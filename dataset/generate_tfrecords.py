import os
import math
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from shutil import rmtree


def split(sequence, n):
  """ divide sequence into n sub-sequence evenly"""
  k, m = divmod(len(sequence), n)
  return [
      sequence[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)
  ]


def normalize(x, x_min, x_max):
  # scale x to be between 0 and 1
  return (x - x_min) / (x_max - x_min)


def get_segments(hparams):
  print('processing file {}...'.format(hparams.input))

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

  num_neurons = raw_signals.shape[0]
  num_samples = raw_signals.shape[1] - hparams.sequence_length
  signals = np.zeros((num_samples, num_neurons, hparams.sequence_length),
                     dtype=np.float32)
  spikes = np.zeros((num_samples, num_neurons, hparams.sequence_length),
                    dtype=np.float32)

  for i in tqdm(range(num_samples)):
    signals[i] = raw_signals[:, i:i + hparams.sequence_length]
    spikes[i] = raw_spikes[:, i:i + hparams.sequence_length]

  assert signals.shape == spikes.shape

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


def write_to_record(hparams, mode, shard, num_shards, signals, spikes):
  record_filename = get_record_filename(hparams, mode, shard, num_shards)
  print('writing {} segments to {}...'.format(len(signals), record_filename))

  with tf.io.TFRecordWriter(record_filename) as writer:
    for signal, spike in zip(signals, spikes):
      example = serialize_example(signal, spike)
      writer.write(example)


def write_to_records(hparams, mode, signals, spikes):
  if not os.path.exists(hparams.output_dir):
    os.makedirs(hparams.output_dir)

  # calculate the number of records to create
  num_shards = 1 if hparams.num_per_shard == 0 else math.ceil(
      len(signals) / hparams.num_per_shard)

  print('writing {} segments to {} {} records...'.format(
      len(signals), num_shards, mode))

  if mode == 'train':
    hparams.num_train_shards = num_shards
  else:
    hparams.num_validation_shards = num_shards

  sharded_signals = split(signals, num_shards)
  sharded_spikes = split(spikes, num_shards)

  for shard in range(num_shards):
    write_to_record(hparams, mode, shard, num_shards, sharded_signals[shard],
                    sharded_spikes[shard])


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
  signals = signals[indexes]
  spikes = spikes[indexes]

  train_size = int(len(signals) * 0.7)

  hparams.train_size = train_size
  hparams.validation_size = len(signals) - train_size
  hparams.signal_shape = signals.shape[1:]
  hparams.spike_shape = spikes.shape[1:]

  write_to_records(
      hparams,
      mode='train',
      signals=signals[:train_size],
      spikes=spikes[:train_size])

  write_to_records(
      hparams,
      mode='validation',
      signals=signals[train_size:],
      spikes=spikes[train_size:])

  # save information of the dataset
  with open(os.path.join(hparams.output_dir, 'info.pkl'), 'wb') as file:
    buffer_size = hparams.num_per_shard if hparams.num_per_shard > 0 else hparams.train_size
    pickle.dump({
        'train_size': hparams.train_size,
        'validation_size': hparams.validation_size,
        'signal_shape': hparams.signal_shape,
        'spike_shape': hparams.spike_shape,
        'num_train_shards': hparams.num_train_shards,
        'num_validation_shards': hparams.num_validation_shards,
        'buffer_size': buffer_size,
        'normalize': hparams.normalize,
        'signals_min': hparams.signals_min,
        'signals_max': hparams.signals_max
    }, file)

  print('saved {} tfrecords to {}'.format(
      hparams.num_train_shards + hparams.num_validation_shards,
      hparams.output_dir))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input', default='raw_data/ST260_Day4_signals4Bryan.pkl', type=str)
  parser.add_argument('--output_dir', default='tfrecords', type=str)
  parser.add_argument('--sequence_length', default=1024, type=int)
  parser.add_argument('--normalize', action='store_true')
  parser.add_argument('--replace', action='store_true')
  parser.add_argument(
      '--num_per_shard',
      default=-1,
      type=int,
      help='number of segments save in each TFRecord file. '
      'if -1 then automatically adjust each shard to be 100MB, '
      'if 0 then save all data to the same file.')
  hparams = parser.parse_args()

  # calculate the number of samples per shard so that each shard is about 100MB
  if hparams.num_per_shard == -1:
    hparams.num_per_shard = int((120 / hparams.sequence_length) * 1100)

  main(hparams)
