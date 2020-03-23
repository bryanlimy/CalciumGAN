import os
import json
import pickle
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from time import time
from multiprocessing import Pool

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


def deconvolve_neuron(hparams, filename, neuron):
  signals = h5_helper.get(filename, name='signals', neuron=neuron)
  signals = utils.set_array_format(signals, data_format='NW', hparams=hparams)
  return spike_helper.deconvolve_signals(signals, threshold=0.5)


def deconvolve_from_file(hparams, filename):
  print('\tDeconvolve {}'.format(filename))

  pool = Pool(hparams.num_processors)
  fake_spikes = pool.starmap(
      deconvolve_neuron,
      [(hparams, filename, n) for n in range(hparams.num_neurons)])
  pool.close()

  fake_spikes = utils.set_array_format(
      np.array(fake_spikes, dtype=np.int8), data_format='NWC', hparams=hparams)

  h5_helper.write(filename, {'spikes': fake_spikes})


def get_neo_trains(hparams,
                   filename,
                   neuron=None,
                   sample=None,
                   data_format=None,
                   num_samples=None):
  assert data_format and (neuron is not None or sample is not None)

  spikes = h5_helper.get(filename, name='spikes', neuron=neuron, sample=sample)
  spikes = utils.set_array_format(spikes, data_format, hparams)

  if num_samples is not None:
    assert data_format[0] == 'N'
    spikes = spikes[:num_samples]

  return spike_helper.trains_to_neo(spikes)


def mse(x, y):
  return np.nanmean(np.square(x - y), dtype=np.float32)


def neuron_firing_rate(hparams, filename, neuron):
  if hparams.verbose == 2:
    print('\tComputing firing rate for neuron #{}'.format(neuron))

  real_spikes = get_neo_trains(
      hparams,
      hparams.validation_cache,
      neuron=neuron,
      data_format='NW',
  )
  fake_spikes = get_neo_trains(
      hparams,
      filename,
      neuron=neuron,
      data_format='NW',
  )

  real_firing_rate = spike_metrics.mean_firing_rate(real_spikes)
  fake_firing_rate = spike_metrics.mean_firing_rate(fake_spikes)
  firing_rate_error = mse(real_firing_rate, fake_firing_rate)

  return {
      'firing_rate_error': firing_rate_error,
      'firing_rate_pair': (real_firing_rate, fake_firing_rate)
  }


def firing_rate_metrics(hparams, summary, filename, epoch):
  if hparams.verbose:
    print('\tComputing firing rate')

  neurons = hparams.focus_neurons if hparams.focus_neurons else range(
      hparams.neurons)

  pool = Pool(hparams.num_processors)
  results = pool.starmap(neuron_firing_rate,
                         [(hparams, filename, n) for n in neurons])
  pool.close()

  firing_rate_errors, firing_rate_pairs = [], []
  for result in results:
    firing_rate_errors.append(result['firing_rate_error'])
    firing_rate_pairs.append(result['firing_rate_pair'])

  summary.scalar(
      'spike_metrics/firing_rate_error',
      np.mean(firing_rate_errors),
      step=epoch,
      training=False)

  summary.plot_histograms(
      'firing_rate_histograms',
      firing_rate_pairs,
      xlabel='Hz',
      ylabel='Count',
      titles=['Neuron #{:03d}'.format(n) for n in neurons],
      step=epoch,
      training=False)


def neuron_covariance(hparams, filename, neuron, num_samples):
  if hparams.verbose == 2:
    print('\t\tComputing covariance for neuron #{}'.format(neuron))

  real_spikes = get_neo_trains(
      hparams,
      hparams.validation_cache,
      neuron=neuron,
      data_format='NW',
      num_samples=num_samples)
  fake_spikes = get_neo_trains(
      hparams,
      filename,
      neuron=neuron,
      data_format='NW',
      num_samples=num_samples)

  return np.mean(spike_metrics.covariance(real_spikes, fake_spikes))


def covariance_metrics(hparams, summary, filename, epoch):
  if hparams.verbose:
    print('\tComputing covariance')

  neurons = hparams.focus_neurons if hparams.focus_neurons else range(
      hparams.num_neurons)

  # compute neuron-wise covariance with 500 samples
  pool = Pool(hparams.num_processors)
  results = pool.starmap(neuron_covariance,
                         [(hparams, filename, n, 500) for n in neurons])
  pool.close()

  summary.scalar(
      'spike_metrics/neuron_covariance',
      np.mean(results),
      step=epoch,
      training=False)


def correlation_coefficient_error(hparams, filename, sample):
  if hparams.verbose == 2:
    print('\t\tComputing correlation coefficient error for sample #{}'.format(
        sample))

  diag_indices = np.triu_indices(hparams.num_neurons, k=1)

  real_spikes = get_neo_trains(
      hparams, hparams.validation_cache, sample=sample, data_format='CW')
  real_corrcoef = spike_metrics.correlation_coefficients(real_spikes, None)
  real_corrcoef = real_corrcoef[diag_indices]

  fake_spikes = get_neo_trains(
      hparams, filename, sample=sample, data_format='CW')
  fake_corrcoef = spike_metrics.correlation_coefficients(fake_spikes, None)
  fake_corrcoef = fake_corrcoef[diag_indices]

  return mse(real_corrcoef, fake_corrcoef)


def correlation_coefficient_sample_histogram(hparams, filename, sample):
  if hparams.verbose == 2:
    print('\t\tComputing correlation coefficient histogram for sample #{}'.
          format(sample))

  diag_indices = np.triu_indices(hparams.num_neurons, k=1)

  real_spikes = get_neo_trains(
      hparams, hparams.validation_cache, sample=sample, data_format='CW')
  real_corrcoef = spike_metrics.correlation_coefficients(real_spikes, None)
  real_corrcoef = utils.remove_nan(real_corrcoef[diag_indices])

  fake_spikes = get_neo_trains(
      hparams, filename, sample=sample, data_format='CW')
  fake_corrcoef = spike_metrics.correlation_coefficients(fake_spikes, None)
  fake_corrcoef = utils.remove_nan(fake_corrcoef[diag_indices])

  return (real_corrcoef, fake_corrcoef)


def correlation_coefficient_samples_mean_histogram(hparams, filename, sample):
  if hparams.verbose == 2:
    print('\t\tComputing mean correlation coefficient histogram for sample #{}'.
          format(sample))

  diag_indices = np.triu_indices(hparams.num_neurons, k=1)

  real_spikes = get_neo_trains(
      hparams, hparams.validation_cache, sample=sample, data_format='CW')
  real_corrcoef = spike_metrics.correlation_coefficients(real_spikes, None)
  real_corrcoef = np.nanmean(real_corrcoef[diag_indices], dtype=np.float32)

  fake_spikes = get_neo_trains(
      hparams, filename, sample=sample, data_format='CW')
  fake_corrcoef = spike_metrics.correlation_coefficients(fake_spikes, None)
  fake_corrcoef = np.nanmean(fake_corrcoef[diag_indices], dtype=np.float32)

  return [real_corrcoef, fake_corrcoef]


def correlation_coefficient_metrics(hparams, summary, filename, epoch):
  if hparams.verbose:
    print('\tComputing correlation coefficient')

  neurons = hparams.focus_neurons if hparams.focus_neurons else range(
      hparams.num_neurons)

  # compute sample-wise covariance for the first 500 samples
  pool = Pool(hparams.num_processors)
  results = pool.starmap(correlation_coefficient_error,
                         [(hparams, filename, i) for i in range(500)])
  pool.close()

  summary.scalar(
      'spike_metrics/correlation_error',
      np.nanmean(results, dtype=np.float32),
      step=epoch,
      training=False)

  # compute sample-wise correlation histogram
  pool = Pool(hparams.num_processors)
  results = pool.starmap(correlation_coefficient_sample_histogram,
                         [(hparams, filename, i) for i in range(10)])
  pool.close()

  summary.plot_histograms(
      'correlation_coefficient_sample_histogram',
      results,
      xlabel='Correlation',
      ylabel='Count',
      titles=['Sample #{:03d}'.format(i) for i in range(len(results))],
      step=epoch,
      training=False)

  # compute mean sample-wise correlation histogram
  pool = Pool(hparams.num_processors)
  results = pool.starmap(correlation_coefficient_samples_mean_histogram,
                         [(hparams, filename, i) for i in range(500)])
  pool.close()

  results = np.array(results, dtype=np.float32)

  summary.plot_histograms(
      'correlation_coefficient_mean_samples_histogram',
      [(results[:, 0], results[:, 1])],
      xlabel='Mean correlation',
      ylabel='Count',
      titles=['Mean correlation over {} samples'.format(len(results))],
      step=epoch,
      training=False)


def neuron_van_rossum_distance(hparams, filename, neuron, num_samples):
  ''' compute van rossum distance for neuron with num_samples samples'''
  if hparams.verbose == 2:
    print('\t\tComputing van-rossum distance for neuron #{}'.format(neuron))

  real_spikes = get_neo_trains(
      hparams,
      hparams.validation_cache,
      neuron=neuron,
      data_format='NW',
      num_samples=num_samples)
  fake_spikes = get_neo_trains(
      hparams,
      filename,
      neuron=neuron,
      data_format='NW',
      num_samples=num_samples)

  return np.mean(spike_metrics.van_rossum_distance(real_spikes, fake_spikes))


def sort_heatmap(matrix):
  ''' sort the given matrix where the top left corner is the minimum'''
  num_samples = len(matrix)

  # create a copy of distances matrix for modification
  matrix_copy = np.copy(matrix)

  heatmap = np.full(matrix.shape, fill_value=np.nan, dtype=np.float32)

  # get the index with the minimum value
  min_index = np.unravel_index(np.argmin(matrix), matrix.shape)

  # row and column order for the sorted matrix
  row_order = np.full((num_samples,), fill_value=-1, dtype=np.int)
  row_order[0] = min_index[0]
  column_order = np.argsort(matrix[min_index[0]])

  for i in range(num_samples):
    if i != 0:
      row_order[i] = np.argsort(matrix_copy[:, column_order[i]])[0]
    heatmap[i] = matrix[row_order[i]][column_order]
    matrix_copy[row_order[i]][:] = np.inf

  return heatmap, row_order, column_order


def sample_van_rossum_heatmap(hparams, filename, sample):
  ''' compute van rossum heatmap for sample '''
  if hparams.verbose == 2:
    print('\t\tComputing van-rossum heatmap for sample #{}'.format(sample))

  real_spikes = get_neo_trains(
      hparams, hparams.validation_cache, sample=sample, data_format='CW')
  fake_spikes = get_neo_trains(
      hparams, filename, sample=sample, data_format='CW')

  distances = spike_metrics.van_rossum_distance(real_spikes, fake_spikes)
  heatmap, row_order, column_order = sort_heatmap(distances)

  return {
      'heatmap': heatmap,
      'xticklabels': row_order,
      'yticklabels': column_order
  }


def neuron_van_rossum_heatmap(hparams, filename, neuron, num_samples):
  ''' compute van rossum heatmap for neuron with num_samples samples '''
  if hparams.verbose == 2:
    print('\t\tComputing van-rossum heatmap for neuron #{}'.format(neuron))

  real_spikes = get_neo_trains(
      hparams,
      hparams.validation_cache,
      neuron=neuron,
      data_format='NW',
      num_samples=num_samples)
  fake_spikes = get_neo_trains(
      hparams,
      filename,
      neuron=neuron,
      data_format='NW',
      num_samples=num_samples)

  distances = spike_metrics.van_rossum_distance(real_spikes, fake_spikes)
  heatmap, row_order, column_order = sort_heatmap(distances)

  return {
      'heatmap': heatmap,
      'xticklabels': row_order,
      'yticklabels': column_order
  }


def sample_van_rossum_histogram(hparams, filename, sample):
  ''' compute van rossum distance for sample '''
  if hparams.verbose == 2:
    print('\t\tComputing van-rossum histograms for sample #{}'.format(sample))

  real_spikes = get_neo_trains(
      hparams,
      hparams.validation_cache,
      sample=sample,
      data_format='CW',
  )
  real_van_rossum = spike_metrics.van_rossum_distance(real_spikes, None)

  fake_spikes = get_neo_trains(
      hparams,
      filename,
      sample=sample,
      data_format='CW',
  )
  fake_van_rossum = spike_metrics.van_rossum_distance(fake_spikes, None)

  assert real_van_rossum.shape == fake_van_rossum.shape

  diag_indices = np.triu_indices(len(real_van_rossum), k=1)

  real_van_rossum = real_van_rossum[diag_indices]
  fake_van_rossum = fake_van_rossum[diag_indices]

  return (real_van_rossum, fake_van_rossum)


def neuron_van_rossum_histogram(hparams, filename, neuron, num_samples):
  ''' compute van rossum distance for neuron '''
  if hparams.verbose == 2:
    print('\t\tComputing van-rossum histograms for neuron #{}'.format(neuron))

  real_spikes = get_neo_trains(
      hparams,
      hparams.validation_cache,
      neuron=neuron,
      data_format='NW',
      num_samples=num_samples)
  real_van_rossum = spike_metrics.van_rossum_distance(real_spikes, None)

  fake_spikes = get_neo_trains(
      hparams,
      filename,
      neuron=neuron,
      data_format='NW',
      num_samples=num_samples)
  fake_van_rossum = spike_metrics.van_rossum_distance(fake_spikes, None)

  diag_indices = np.triu_indices(len(real_van_rossum), k=1)

  real_van_rossum = real_van_rossum[diag_indices]
  fake_van_rossum = fake_van_rossum[diag_indices]

  return (real_van_rossum, fake_van_rossum)


def van_rossum_metrics(hparams, summary, filename, epoch):
  if hparams.verbose:
    print('\tComputing van-rossum distance')

  neurons = hparams.focus_neurons if hparams.focus_neurons else range(
      hparams.num_neurons)

  # compute neuron-wise van rossum distance error with 500 samples
  pool = Pool(hparams.num_processors)
  results = pool.starmap(neuron_van_rossum_distance,
                         [(hparams, filename, n, 500) for n in neurons])
  pool.close()

  summary.scalar(
      'spike_metrics/van_rossum_neuron',
      np.mean(results),
      step=epoch,
      training=False)

  # compute sample-wise van rossum heat-map
  pool = Pool(hparams.num_processors)
  results = pool.starmap(sample_van_rossum_heatmap,
                         [(hparams, filename, i) for i in range(10)])
  pool.close()

  heatmaps, xticklabels, yticklabels, titles = [], [], [], []
  for i in range(len(results)):
    heatmaps.append(results[i]['heatmap'])
    xticklabels.append(results[i]['xticklabels'])
    yticklabels.append(results[i]['yticklabels'])
    titles.append('Sample #{:03d}'.format(i))

  summary.plot_heatmaps(
      'van_rossum_sample_heatmaps',
      heatmaps,
      xlabel='Fake neuron',
      ylabel='Real neuron',
      xticklabels=xticklabels,
      yticklabels=yticklabels,
      titles=titles,
      step=epoch,
      training=False)

  # compute neuron-wise van rossum heat-map for 100 samples
  pool = Pool(hparams.num_processors)
  results = pool.starmap(neuron_van_rossum_heatmap,
                         [(hparams, filename, i, 100) for i in neurons])
  pool.close()

  heatmaps, xticklabels, yticklabels, titles = [], [], [], []
  for i in range(len(results)):
    heatmaps.append(results[i]['heatmap'])
    xticklabels.append(results[i]['xticklabels'])
    yticklabels.append(results[i]['yticklabels'])
    titles.append('Neuron #{:03d}'.format(neurons[i]))

  summary.plot_heatmaps(
      'van_rossum_neuron_heatmaps',
      heatmaps,
      xlabel='Fake sample',
      ylabel='Real sample',
      xticklabels=xticklabels,
      yticklabels=yticklabels,
      titles=titles,
      step=epoch,
      training=False)

  # compute sample-wise van rossum distance histogram
  pool = Pool(hparams.num_processors)
  results = pool.starmap(sample_van_rossum_histogram,
                         [(hparams, filename, i) for i in range(10)])
  pool.close()

  summary.plot_histograms(
      'van_rossum_sample_histograms',
      results,
      xlabel='Distance',
      ylabel='Count',
      titles=['Sample #{:03d}'.format(i) for i in range(len(results))],
      step=epoch,
      training=False)

  # compute neuron-wise van rossum distance histogram for 500 samples
  pool = Pool(hparams.num_processors)
  results = pool.starmap(neuron_van_rossum_histogram,
                         [(hparams, filename, n, 500) for n in neurons])
  pool.close()

  summary.plot_histograms(
      'van_rossum_neuron_histograms',
      results,
      xlabel='Distance',
      ylabel='Count',
      titles=['Neuron #{:03d}'.format(i) for i in neurons],
      step=epoch,
      training=False)


def compute_epoch_spike_metrics(hparams, summary, filename, epoch):
  if not h5_helper.contains(filename, 'spikes'):
    deconvolve_from_file(hparams, filename)

  firing_rate_metrics(hparams, summary, filename, epoch)

  # covariance_metrics(hparams, summary, filename, epoch)

  correlation_coefficient_metrics(hparams, summary, filename, epoch)

  van_rossum_metrics(hparams, summary, filename, epoch)


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
    compute_epoch_spike_metrics(
        hparams, summary, filename=info[epoch]['filename'], epoch=epoch)
    end = time()

    summary.scalar(
        'elapse/spike_metrics', end - start, step=epoch, training=False)

    print('{} took {:.02f} mins'.format(info[epoch]['filename'],
                                        (end - start) / 60))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', default='runs')
  parser.add_argument('--num_processors', default=6, type=int)
  parser.add_argument('--focus_neurons', action='store_true')
  parser.add_argument('--verbose', default=1, type=int)
  hparams = parser.parse_args()

  warnings.simplefilter(action='ignore', category=UserWarning)
  warnings.simplefilter(action='ignore', category=RuntimeWarning)
  warnings.simplefilter(action='ignore', category=DeprecationWarning)

  # hand picked neurons to plots
  if hparams.focus_neurons:
    hparams.focus_neurons = [87, 58, 90, 39, 7, 60, 14, 5, 13]

  main(hparams)
