import os
import json
import pickle
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from glob import glob
from time import time
import tensorflow as tf
from multiprocessing import Pool, Manager

from gan.utils import utils
from gan.utils import h5_helper
from gan.utils import spike_metrics
from gan.utils import spike_helper
from gan.utils.summary_helper import Summary


def check_path_exists(path):
  if not os.path.exists(path):
    print('{} not found'.format(path))
    exit()


def load_hparams(hparams):
  filename = os.path.join(hparams.output_dir, 'hparams.json')
  with open(filename, 'r') as file:
    content = json.load(file)
  for key, value in content.items():
    if not hasattr(hparams, key):
      setattr(hparams, key, value)


def load_info(hparams):
  filename = os.path.join(hparams.generated_dir, 'info.pkl')
  with open(filename, 'rb') as file:
    info = pickle.load(file)
  return info


def deconvolve_from_file(hparams, filename):
  fake_signals = h5_helper.get(filename, name='signals')
  pool = Pool(hparams.num_processors)
  fake_spikes = pool.starmap(
      spike_helper.deconvolve_signals,
      [(fake_signals[i], 0.5, False) for i in range(len(fake_signals))])
  pool.close()

  fake_spikes = np.array(fake_spikes, dtype=np.float32)
  h5_helper.write(filename, {'spikes': fake_spikes})


def get_neo_trains(filename, hparams, index, neuron):
  # get real neuron data
  real_spikes = h5_helper.get(
      filename, name='spikes', index=index, neuron=neuron, hparams=hparams)
  return spike_helper.trains_to_neo(real_spikes)


def mean_firing_rate(hparams, filename, neuron):
  if hparams.verbose == 2:
    print('\tComputing firing rate for neuron #{}'.format(neuron))

  real_spikes = get_neo_trains(
      hparams.validation_cache, hparams, index=neuron, neuron=True)
  fake_spikes = get_neo_trains(filename, hparams, index=neuron, neuron=True)

  real_firing_rate = spike_metrics.mean_firing_rate(real_spikes)
  fake_firing_rate = spike_metrics.mean_firing_rate(fake_spikes)
  firing_rate_error = np.mean(np.square(real_firing_rate - fake_firing_rate))
  return {
      'firing_rate_error': firing_rate_error,
      'firing_rate_pair': (real_firing_rate, fake_firing_rate)
  }


def firing_rate_metrics(hparams, info, summary):
  if hparams.verbose:
    print('\tComputing firing rate')

  pool = Pool(hparams.num_processors)
  results = pool.starmap(
      mean_firing_rate,
      [(hparams, info['filename'], n) for n in range(hparams.num_neurons)])
  pool.close()

  firing_rate_errors, firing_rate_pairs = [], []
  for result in results:
    firing_rate_errors.append(result['firing_rate_error'])
    firing_rate_pairs.append(result['firing_rate_pair'])

  summary.scalar(
      'spike_metrics/firing_rate_error',
      np.mean(firing_rate_errors),
      step=info['global_step'],
      training=False)

  summary.plot_histograms(
      'firing_rate_histograms',
      firing_rate_pairs,
      xlabel='Hz',
      ylabel='Count',
      title='Neuron #{:03d}',
      step=info['global_step'],
      training=False)


def covariance(hparams, filename, neuron):
  if hparams.verbose == 2:
    print('\t\tComputing covariance for neuron #{}'.format(neuron))

  real_spikes = get_neo_trains(
      hparams.validation_cache, hparams, index=neuron, neuron=True)
  fake_spikes = get_neo_trains(filename, hparams, index=neuron, neuron=True)

  return np.mean(spike_metrics.covariance(real_spikes, fake_spikes))


def covariance_metrics(hparams, info, summary):
  if hparams.verbose:
    print('\tComputing covariance')

  pool = Pool(hparams.num_processors)
  results = pool.starmap(
      covariance,
      [(hparams, info['filename'], n) for n in range(hparams.num_neurons)])
  pool.close()

  summary.scalar(
      'spike_metrics/covariance',
      np.mean(results),
      step=info['global_step'],
      training=False)


def neuron_van_rossum_distance(hparams, filename, neuron):
  if hparams.verbose == 2:
    print('\t\tComputing van-rossum distance for neuron #{}'.format(neuron))

  real_spikes = get_neo_trains(
      hparams.validation_cache, hparams, index=neuron, neuron=True)[:500]
  fake_spikes = get_neo_trains(
      filename, hparams, index=neuron, neuron=True)[:500]

  return np.mean(spike_metrics.van_rossum_distance(real_spikes, fake_spikes))


def sample_van_rossum_histogram(hparams, filename, sample):
  if hparams.verbose == 2:
    print('\t\tComputing van-rossum distance for sample #{}'.format(sample))

  real_spikes = get_neo_trains(
      hparams.validation_cache, hparams, index=sample, neuron=False)
  real_van_rossum = spike_metrics.van_rossum_distance(real_spikes, None)

  fake_spikes = get_neo_trains(filename, hparams, index=sample, neuron=False)
  fake_van_rossum = spike_metrics.van_rossum_distance(fake_spikes, None)

  assert real_van_rossum.shape == fake_van_rossum.shape

  diag_indices = np.triu_indices(len(real_van_rossum), k=1)

  real_van_rossum = real_van_rossum[diag_indices]
  fake_van_rossum = fake_van_rossum[diag_indices]

  return (real_van_rossum, fake_van_rossum)


def van_rossum_metrics(hparams, info, summary):
  if hparams.verbose:
    print('\tComputing van-rossum distance')

  pool = Pool(hparams.num_processors)
  results = pool.starmap(
      neuron_van_rossum_distance,
      [(hparams, info['filename'], n) for n in range(hparams.num_neurons)])
  pool.close()

  summary.scalar(
      'spike_metrics/van_rossum_distance',
      np.mean(results),
      step=info['global_step'],
      training=False)

  # get the first 100 samples van rossum distance
  pool = Pool(hparams.num_processors)
  results = pool.starmap(sample_van_rossum_histogram,
                         [(hparams, info['filename'], i) for i in range(100)])
  pool.close()

  summary.plot_histograms(
      'van_rossum_distance_histograms',
      results,
      xlabel='Distance',
      ylabel='Count',
      title='Sample #{:03d}',
      step=info['global_step'],
      training=False)


def compute_epoch_spike_metrics(hparams, info, summary):
  if not h5_helper.contains(info['filename'], 'spikes'):
    deconvolve_from_file(hparams, info['filename'])

  firing_rate_metrics(hparams, info, summary)

  covariance_metrics(hparams, info, summary)

  van_rossum_metrics(hparams, info, summary)


def main(hparams):
  check_path_exists(hparams.output_dir)

  load_hparams(hparams)
  info = load_info(hparams)
  summary = Summary(hparams)

  epochs = sorted(list(info.keys()))
  for epoch in epochs:
    start = time()
    if hparams.verbose:
      print('\nCompute metrics for {}'.format(info[epoch]['filename']))
    compute_epoch_spike_metrics(hparams, info[epoch], summary)
    end = time()
    print('{} took {:.02f}s'.format(info[epoch]['filename'], end - start))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', default='runs')
  parser.add_argument(
      '--num_processors',
      default=8,
      type=int,
      help='number of processing cores to use for metrics calculation')
  parser.add_argument('--verbose', default=1, type=int)
  hparams = parser.parse_args()

  if hparams.verbose != 2:
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

  main(hparams)
