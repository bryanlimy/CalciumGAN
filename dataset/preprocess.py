import os
import math
import pickle
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from sklearn.preprocessing import normalize
from multiprocessing import Process, cpu_count


def split(sequence, n):
  """ divide sequence into n sub-sequence evenly"""
  k, m = divmod(len(sequence), n)
  return [
      sequence[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)
  ]


def get_segments_from_file(hparams, filename):
  print('processing file {}...'.format(filename))
  with open(filename, 'rb') as file:
    data = pickle.load(file)

  raw_signals = np.array(data['signals'], dtype=np.float32)
  raw_spikes = np.array(data['oasis'], dtype=np.float32)

  # remove first two rows in signals
  raw_signals = raw_signals[2:]
  raw_spikes = raw_spikes[2:]

  assert raw_signals.shape == raw_spikes.shape

  signals, spikes = [], []
  for i in tqdm(range(raw_signals.shape[-1] - hparams.sequence_length)):
    signals.append(raw_signals[:, i:i + hparams.sequence_length])
    spikes.append(raw_spikes[:, i:i + hparams.sequence_length])

  signals = np.concatenate(signals, axis=0)
  spikes = np.concatenate(spikes, axis=0)

  assert signals.shape == spikes.shape

  return signals, spikes


def get_segments(hparams):
  filenames = glob(os.path.join(hparams.input_dir, '*.pkl'))
  filenames.sort()

  signals, spikes = [], []
  for filename in filenames:
    signal, spike = get_segments_from_file(hparams, filename)
    signals.append(signal)
    spikes.append(spike)

  signals = np.concatenate(signals, axis=0)
  spikes = np.concatenate(spikes, axis=0)

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


def write_to_record(hparams, mode, shard_ids, num_shards, signals, spikes):
  for i in range(len(shard_ids)):
    record_filename = get_record_filename(hparams, mode, shard_ids[i],
                                          num_shards)
    print('writing {} segments to {}...'.format(
        len(signals[i]), record_filename))

    with tf.io.TFRecordWriter(record_filename) as writer:
      for signal, spike in zip(signals[i], spikes[i]):
        example = serialize_example(signal, spike)
        writer.write(example)


def write_to_records(hparams, mode, signals, spikes):
  if not os.path.exists(hparams.output_dir):
    os.makedirs(hparams.output_dir)
  # calculate the number of records to create
  num_shards = math.ceil(len(signals) / hparams.num_per_shard)

  print('writing {} segments to {} {} records...'.format(
      len(signals), num_shards, mode))

  if mode == 'train':
    hparams.train_shards = num_shards
  else:
    hparams.eval_shards = num_shards

  signals, spikes = split(signals, num_shards), split(spikes, num_shards)

  num_jobs = min(len(signals), hparams.num_jobs)

  signals, spikes = split(signals, num_jobs), split(spikes, num_jobs)
  shard_ids = split(list(range(num_shards)), num_jobs)

  jobs = []
  for i in range(num_jobs):
    process = Process(
        target=write_to_record,
        args=(hparams, mode, shard_ids[i], num_shards, signals[i], spikes[i]))
    jobs.append(process)
    process.start()

  for process in jobs:
    process.join()


def get_mean_spike(spikes):
  binarized = (spikes > np.random.random(spikes.shape)).astype(np.float32)
  return np.mean(binarized)


def main(hparams):
  if not os.path.exists(hparams.input_dir):
    print('input directory {} does not exists'.format(hparams.input_dir))
    exit()

  if os.path.exists(hparams.output_dir):
    print('output directory {} already exists\n'.format(hparams.output_dir))
    exit()

  signals, spikes = get_segments(hparams)

  if hparams.normalize:
    print('apply {} normalization'.format(hparams.normalize))
    signals = normalize(signals, norm=hparams.normalize, axis=1)

  # shuffle data
  indexes = np.arange(len(signals))
  np.random.shuffle(indexes)
  signals = signals[indexes]
  spikes = spikes[indexes]

  train_size = int(len(signals) * 0.7)

  hparams.train_size = train_size
  hparams.eval_size = len(signals) - train_size
  hparams.signal_shape = signals.shape[1:]
  hparams.spike_shape = spikes.shape[1:]

  hparams.mean_spike_count = get_mean_spike(spikes[train_size:])

  write_to_records(
      hparams,
      mode='train',
      signals=signals[:train_size],
      spikes=spikes[:train_size])

  write_to_records(
      hparams,
      mode='eval',
      signals=signals[train_size:],
      spikes=spikes[train_size:])

  # save information of the dataset
  with open(os.path.join(hparams.output_dir, 'info.pkl'), 'wb') as file:
    pickle.dump({
        'train_size': hparams.train_size,
        'eval_size': hparams.eval_size,
        'signal_shape': hparams.signal_shape,
        'spike_shape': hparams.spike_shape,
        'train_shards': hparams.train_shards,
        'eval_shards': hparams.eval_shards,
        'num_per_shard': hparams.num_per_shard,
        'normalize': hparams.normalize,
        'mean_spike_count': hparams.mean_spike_count
    }, file)

  print('saved {} tfrecords to {}'.format(
      hparams.train_shards + hparams.eval_shards, hparams.output_dir))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', default='raw_data', type=str)
  parser.add_argument('--output_dir', default='tfrecords', type=str)
  parser.add_argument('--sequence_length', default=120, type=int)
  parser.add_argument('--num_per_shard', default=100000, type=int)
  parser.add_argument('--num_jobs', default=cpu_count(), type=int)
  parser.add_argument(
      '--normalize', default='', type=str, choices=['', 'l1', 'l2', 'max'])
  hparams = parser.parse_args()
  main(hparams)
