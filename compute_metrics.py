import os
import json
import pickle
import argparse
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


def _firing_rate_metrics(hparams, filename, neuron):
  if hparams.verbose == 2:
    print('\tComputing firing rate for neuron #{}'.format(neuron))
  # get real neuron data
  real_spikes = h5_helper.get(
      hparams.validation_cache,
      name='spikes',
      index=neuron,
      neuron=True,
      hparams=hparams)
  real_spikes = spike_helper.trains_to_neo(real_spikes)

  # get fake neuron data
  fake_spikes = h5_helper.get(
      filename, name='spikes', index=neuron, neuron=True, hparams=hparams)
  fake_spikes = spike_helper.trains_to_neo(fake_spikes)

  real_firing_rate = spike_metrics.mean_firing_rate(real_spikes)
  fake_firing_rate = spike_metrics.mean_firing_rate(fake_spikes)
  firing_rate_error = np.mean(np.square(real_firing_rate - fake_firing_rate))
  return {
      'firing_rate_error': firing_rate_error,
      'firing_rate_pair': (real_firing_rate, fake_firing_rate)
  }


def firing_rate_metrics(hparams, info, summary):
  if hparams.verbose:
    print('Computing firing rate')

  pool = Pool(hparams.num_processors)
  results = pool.starmap(
      _firing_rate_metrics,
      [(hparams, info['filename'], n) for n in range(hparams.num_neurons)])
  pool.close()

  firing_rate_errors, firing_rate_pairs = [], []
  for result in results:
    firing_rate_errors.append(result['firing_rate_error'])
    firing_rate_pairs.append(results['firing_rate_pair'])

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
      step=info['global_step'],
      training=False)


def correlation_coefficients_metrics(metrics, neuron, real_spikes, fake_spikes):
  corrcoef = spike_metrics.correlation_coefficients(real_spikes, fake_spikes)
  metrics['spike_metrics/cross_coefficient'][neuron] = np.mean(corrcoef)


def covariance_metrics(metrics, neuron, real_spikes, fake_spikes):
  covariance = spike_metrics.covariance(real_spikes, fake_spikes)
  metrics['spike_metrics/covariance'][neuron] = np.mean(covariance)


def van_rossum_metrics(metrics, neuron, real_spikes, fake_spikes):
  distance = spike_metrics.van_rossum_distance(real_spikes[:1000],
                                               fake_spikes[:1000])
  metrics['spike_metrics/van_rossum_distance'][neuron] = np.mean(distance)

  if 'histogram/van_rossum_distance' in metrics:
    metrics['histogram/van_rossum_distance'][neuron] = (distance, [])

  if 'heatmap/van_rossum_distance' in metrics:
    metrics['heatmap/van_rossum_distance'][neuron] = distance


def compute_epoch_spike_metrics(hparams, info, summary):
  if not h5_helper.contains(info['filename'], 'spikes'):
    deconvolve_from_file(hparams, info['filename'])

  firing_rate_metrics(hparams, info, summary)


def main(hparams):
  check_path_exists(hparams.output_dir)

  load_hparams(hparams)
  info = load_info(hparams)
  summary = Summary(hparams)

  epochs = sorted(list(info.keys()))
  for epoch in epochs:
    if hparams.verbose:
      print('Compute metrics for {}'.format(info[epoch]['filename']))
    compute_epoch_spike_metrics(hparams, info[epoch], summary)


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

  main(hparams)
