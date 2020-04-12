import os
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from multiprocessing import Pool

# use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from gan.utils import utils
from gan.utils import h5_helper
from gan.utils import spike_metrics
from gan.utils import spike_helper
from gan.utils.summary_helper import Summary


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


def raster_plots(hparams, summary, filename, epoch):
  real_spikes = h5_helper.get(hparams.validation_cache, name='spikes')
  real_spikes = utils.set_array_format(real_spikes, 'NCW', hparams)
  fake_spikes = h5_helper.get(filename, name='spikes')
  fake_spikes = utils.set_array_format(fake_spikes, 'NCW', hparams)

  summary.raster_plot(
      'raster_plot_trial',
      real_spikes=real_spikes[0],
      fake_spikes=fake_spikes[0],
      xlabel='Time (ms)',
      ylabel='Neuron',
      title='Trial #001',
      step=epoch,
      training=False)

  summary.raster_plot(
      'raster_plot_neuron',
      real_spikes=real_spikes[:100, 5, :],
      fake_spikes=fake_spikes[:100, 5, :],
      xlabel='Time (ms)',
      ylabel='Trial',
      title='Neuron #005',
      step=epoch,
      training=False)


def get_neo_trains(hparams,
                   filename,
                   neuron=None,
                   trial=None,
                   data_format=None,
                   num_trials=None):
  assert data_format and (neuron is not None or trial is not None)

  spikes = h5_helper.get(filename, name='spikes', neuron=neuron, trial=trial)
  spikes = utils.set_array_format(spikes, data_format, hparams)

  if num_trials is not None:
    assert data_format[0] == 'N'
    spikes = spikes[:num_trials]

  return spike_helper.trains_to_neo(spikes)


def mse(x, y):
  return np.nanmean(np.square(x - y), dtype=np.float32)


def firing_rate_neuron(hparams, filename, neuron):
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


def firing_rate_kl(firing_rate_pairs):

  def kl_divergence(p, q):
    return np.sum(
        np.where(
            np.logical_or(np.equal(p, 0), np.equal(q, 0)),
            0,
            p * np.log(p / q),
        ))

  kl = []
  for i in range(len(firing_rate_pairs)):
    real_firing_rate, fake_firing_rate = firing_rate_pairs[i]
    assert real_firing_rate.shape == fake_firing_rate.shape

    df = pd.DataFrame({
        'firing_rate':
        np.concatenate([real_firing_rate, fake_firing_rate]),
        'real_data':
        [True] * len(real_firing_rate) + [False] * len(fake_firing_rate)
    })

    num_bins = 30
    df['bins'] = pd.cut(
        df.firing_rate, bins=num_bins, labels=np.arange(num_bins))

    real_pdf = np.array([
        len(df[(df.bins == i) & (df.real_data == True)])
        for i in range(num_bins)
    ],
                        dtype=np.float32) / len(real_firing_rate)
    fake_pdf = np.array([
        len(df[(df.bins == i) & (df.real_data == False)])
        for i in range(num_bins)
    ],
                        dtype=np.float32) / len(fake_firing_rate)

    kl.append(kl_divergence(real_pdf, fake_pdf))
  return kl


def firing_rate_metrics(hparams, summary, filename, epoch):
  if hparams.verbose:
    print('\tComputing firing rate')

  pool = Pool(hparams.num_processors)
  results = pool.starmap(firing_rate_neuron,
                         [(hparams, filename, n) for n in hparams.neurons])
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

  summary.plot_histograms_grid(
      'firing_rate_histograms',
      firing_rate_pairs,
      xlabel='Hz',
      ylabel='Count',
      titles=['Neuron #{:03d}'.format(n) for n in hparams.neurons],
      step=epoch,
      training=False)

  # get firing rate for all neurons
  pool = Pool(hparams.num_processors)
  results = pool.starmap(
      firing_rate_neuron,
      [(hparams, filename, n) for n in range(hparams.num_neurons)])
  pool.close()

  firing_rate_pairs = [result['firing_rate_pair'] for result in results]
  kl = firing_rate_kl(firing_rate_pairs)

  summary.plot_distribution(
      'firing_rate_kl_histogram',
      kl,
      xlabel='KL divergence',
      ylabel='Count',
      title='KL divergence histogram of {} neurons'.format(
          len(firing_rate_pairs)),
      step=epoch,
      training=False)


def neuron_covariance(hparams, filename, neuron, num_trials):
  if hparams.verbose == 2:
    print('\t\tComputing covariance for neuron #{}'.format(neuron))

  real_spikes = get_neo_trains(
      hparams,
      hparams.validation_cache,
      neuron=neuron,
      data_format='NW',
      num_trials=num_trials)
  fake_spikes = get_neo_trains(
      hparams, filename, neuron=neuron, data_format='NW', num_trials=num_trials)

  return np.mean(spike_metrics.covariance(real_spikes, fake_spikes))


def covariance_metrics(hparams, summary, filename, epoch):
  if hparams.verbose:
    print('\tComputing covariance')

  # compute neuron-wise covariance with 200 trials
  pool = Pool(hparams.num_processors)
  results = pool.starmap(neuron_covariance,
                         [(hparams, filename, n, 200) for n in hparams.neurons])
  pool.close()

  summary.scalar(
      'spike_metrics/neuron_covariance',
      np.mean(results),
      step=epoch,
      training=False)


def correlation_coefficient_error(hparams, filename, trial):
  if hparams.verbose == 2:
    print('\t\tComputing correlation coefficient error for trial #{}'.format(
        trial))

  diag_indices = np.triu_indices(hparams.num_neurons, k=1)

  real_spikes = get_neo_trains(
      hparams, hparams.validation_cache, trial=trial, data_format='CW')
  real_corrcoef = spike_metrics.correlation_coefficients(real_spikes, None)
  real_corrcoef = real_corrcoef[diag_indices]

  fake_spikes = get_neo_trains(hparams, filename, trial=trial, data_format='CW')
  fake_corrcoef = spike_metrics.correlation_coefficients(fake_spikes, None)
  fake_corrcoef = fake_corrcoef[diag_indices]

  return mse(real_corrcoef, fake_corrcoef)


def correlation_coefficient_trial_histogram(hparams, filename, trial):
  if hparams.verbose == 2:
    print('\t\tComputing correlation coefficient histogram for sample #{}'.
          format(trial))

  diag_indices = np.triu_indices(hparams.num_neurons, k=1)

  real_spikes = get_neo_trains(
      hparams, hparams.validation_cache, trial=trial, data_format='CW')
  real_corrcoef = spike_metrics.correlation_coefficients(real_spikes, None)
  real_corrcoef = utils.remove_nan(real_corrcoef[diag_indices])

  fake_spikes = get_neo_trains(hparams, filename, trial=trial, data_format='CW')
  fake_corrcoef = spike_metrics.correlation_coefficients(fake_spikes, None)
  fake_corrcoef = utils.remove_nan(fake_corrcoef[diag_indices])

  return (real_corrcoef, fake_corrcoef)


def correlation_coefficient_trials_mean_histogram(hparams, filename, trial):
  if hparams.verbose == 2:
    print('\t\tComputing mean correlation coefficient histogram for trial #{}'.
          format(trial))

  diag_indices = np.triu_indices(hparams.num_neurons, k=1)

  real_spikes = get_neo_trains(
      hparams, hparams.validation_cache, trial=trial, data_format='CW')
  real_corrcoef = spike_metrics.correlation_coefficients(real_spikes, None)
  real_corrcoef = np.nanmean(real_corrcoef[diag_indices], dtype=np.float32)

  fake_spikes = get_neo_trains(hparams, filename, trial=trial, data_format='CW')
  fake_corrcoef = spike_metrics.correlation_coefficients(fake_spikes, None)
  fake_corrcoef = np.nanmean(fake_corrcoef[diag_indices], dtype=np.float32)

  return [real_corrcoef, fake_corrcoef]


def correlation_coefficient_metrics(hparams, summary, filename, epoch):
  if hparams.verbose:
    print('\tComputing correlation coefficient')

  # compute trial-wise covariance for the first 200 trials
  pool = Pool(hparams.num_processors)
  results = pool.starmap(correlation_coefficient_error,
                         [(hparams, filename, i) for i in range(200)])
  pool.close()

  summary.scalar(
      'spike_metrics/correlation_error',
      np.nanmean(results, dtype=np.float32),
      step=epoch,
      training=False)

  # compute sample-wise correlation histogram
  pool = Pool(hparams.num_processors)
  results = pool.starmap(correlation_coefficient_trial_histogram,
                         [(hparams, filename, i) for i in range(9)])
  pool.close()

  summary.plot_histograms_grid(
      'correlation_coefficient_trial_histogram',
      results,
      xlabel='Correlation',
      ylabel='Count',
      titles=['Trial #{:03d}'.format(i) for i in range(len(results))],
      step=epoch,
      training=False)

  # compute mean trial-wise correlation histogram
  pool = Pool(hparams.num_processors)
  results = pool.starmap(correlation_coefficient_trials_mean_histogram,
                         [(hparams, filename, i) for i in range(200)])
  pool.close()

  results = np.array(results, dtype=np.float32)

  summary.plot_histogram(
      'correlation_coefficient_mean_trials_histogram',
      (results[:, 0], results[:, 1]),
      xlabel='Mean correlation',
      ylabel='Count',
      title='Mean correlation over {} trials'.format(len(results)),
      step=epoch,
      training=False)


def neuron_van_rossum_distance(hparams, filename, neuron, num_trials):
  ''' compute van rossum distance for neuron with num_trials'''
  if hparams.verbose == 2:
    print('\t\tComputing van-rossum distance for neuron #{}'.format(neuron))

  real_spikes = get_neo_trains(
      hparams,
      hparams.validation_cache,
      neuron=neuron,
      data_format='NW',
      num_trials=num_trials)
  fake_spikes = get_neo_trains(
      hparams, filename, neuron=neuron, data_format='NW', num_trials=num_trials)

  return np.mean(spike_metrics.van_rossum_distance(real_spikes, fake_spikes))


def sort_heatmap(matrix):
  ''' sort the given matrix where the top left corner is the minimum'''
  num_trials = len(matrix)

  # create a copy of distances matrix for modification
  matrix_copy = np.copy(matrix)

  heatmap = np.full(matrix.shape, fill_value=np.nan, dtype=np.float32)

  # get the index with the minimum value
  min_index = np.unravel_index(np.argmin(matrix), matrix.shape)

  # row and column order for the sorted matrix
  row_order = np.full((num_trials,), fill_value=-1, dtype=np.int)
  row_order[0] = min_index[0]
  column_order = np.argsort(matrix[min_index[0]])

  for i in range(num_trials):
    if i != 0:
      row_order[i] = np.argsort(matrix_copy[:, column_order[i]])[0]
    heatmap[i] = matrix[row_order[i]][column_order]
    matrix_copy[row_order[i]][:] = np.inf

  return heatmap, row_order, column_order


def van_rossum_trial_heatmap(hparams, filename, trial):
  ''' compute van rossum heatmap for trial '''
  if hparams.verbose == 2:
    print('\t\tComputing van-rossum heatmap for trial #{}'.format(trial))

  real_spikes = get_neo_trains(
      hparams, hparams.validation_cache, trial=trial, data_format='CW')
  fake_spikes = get_neo_trains(hparams, filename, trial=trial, data_format='CW')

  distances = spike_metrics.van_rossum_distance(real_spikes, fake_spikes)
  heatmap, row_order, column_order = sort_heatmap(distances)

  return {
      'heatmap': heatmap,
      'xticklabels': row_order,
      'yticklabels': column_order
  }


def van_rossum_neuron_heatmap(hparams, filename, neuron, num_trials):
  ''' compute van rossum heatmap for neuron with num_trials '''
  if hparams.verbose == 2:
    print('\t\tComputing van-rossum heatmap for neuron #{}'.format(neuron))

  real_spikes = get_neo_trains(
      hparams,
      hparams.validation_cache,
      neuron=neuron,
      data_format='NW',
      num_trials=num_trials)
  fake_spikes = get_neo_trains(
      hparams, filename, neuron=neuron, data_format='NW', num_trials=num_trials)

  distances = spike_metrics.van_rossum_distance(real_spikes, fake_spikes)
  heatmap, row_order, column_order = sort_heatmap(distances)

  return {
      'heatmap': heatmap,
      'xticklabels': row_order,
      'yticklabels': column_order
  }


def van_rossum_trial_histogram(hparams, filename, trial):
  ''' compute van rossum distance for trial '''
  if hparams.verbose == 2:
    print('\t\tComputing van-rossum histograms for trial #{}'.format(trial))

  real_spikes = get_neo_trains(
      hparams,
      hparams.validation_cache,
      trial=trial,
      data_format='CW',
  )
  real_van_rossum = spike_metrics.van_rossum_distance(real_spikes, None)

  fake_spikes = get_neo_trains(
      hparams,
      filename,
      trial=trial,
      data_format='CW',
  )
  fake_van_rossum = spike_metrics.van_rossum_distance(fake_spikes, None)

  assert real_van_rossum.shape == fake_van_rossum.shape

  diag_indices = np.triu_indices(len(real_van_rossum), k=1)

  real_van_rossum = real_van_rossum[diag_indices]
  fake_van_rossum = fake_van_rossum[diag_indices]

  return (real_van_rossum, fake_van_rossum)


def van_rossum_neuron_histogram(hparams, filename, neuron, num_trials):
  ''' compute van rossum distance for neuron '''
  if hparams.verbose == 2:
    print('\t\tComputing van-rossum histograms for neuron #{}'.format(neuron))

  real_spikes = get_neo_trains(
      hparams,
      hparams.validation_cache,
      neuron=neuron,
      data_format='NW',
      num_trials=num_trials)
  real_van_rossum = spike_metrics.van_rossum_distance(real_spikes, None)

  fake_spikes = get_neo_trains(
      hparams, filename, neuron=neuron, data_format='NW', num_trials=num_trials)
  fake_van_rossum = spike_metrics.van_rossum_distance(fake_spikes, None)

  diag_indices = np.triu_indices(len(real_van_rossum), k=1)

  real_van_rossum = real_van_rossum[diag_indices]
  fake_van_rossum = fake_van_rossum[diag_indices]

  return (real_van_rossum, fake_van_rossum)


def van_rossum_metrics(hparams, summary, filename, epoch):
  if hparams.verbose:
    print('\tComputing van-rossum distance')

  # compute neuron-wise van rossum distance error with 200 trials
  pool = Pool(hparams.num_processors)
  results = pool.starmap(neuron_van_rossum_distance,
                         [(hparams, filename, n, 200) for n in hparams.neurons])
  pool.close()

  summary.scalar(
      'spike_metrics/van_rossum_neuron',
      np.mean(results),
      step=epoch,
      training=False)

  # compute trial-wise van rossum heat-map
  pool = Pool(hparams.num_processors)
  results = pool.starmap(van_rossum_trial_heatmap,
                         [(hparams, filename, i) for i in range(12)])
  pool.close()

  heatmaps, xticklabels, yticklabels, titles = [], [], [], []
  for i in range(len(results)):
    heatmaps.append(results[i]['heatmap'])
    xticklabels.append(results[i]['xticklabels'])
    yticklabels.append(results[i]['yticklabels'])
    titles.append('Trial #{:03d}'.format(i))

  summary.plot_heatmaps_grid(
      'van_rossum_trial_heatmaps',
      heatmaps,
      xlabel='Fake neurons',
      ylabel='Real neurons',
      xticklabels=xticklabels,
      yticklabels=yticklabels,
      titles=titles,
      step=epoch,
      training=False)

  # compute neuron-wise van rossum heat-map for 50 trials
  pool = Pool(hparams.num_processors)
  results = pool.starmap(van_rossum_neuron_heatmap,
                         [(hparams, filename, n, 50) for n in hparams.neurons])
  pool.close()

  heatmaps, xticklabels, yticklabels, titles = [], [], [], []
  for i in range(len(results)):
    heatmaps.append(results[i]['heatmap'])
    xticklabels.append(results[i]['xticklabels'])
    yticklabels.append(results[i]['yticklabels'])
    titles.append('Neuron #{:03d}'.format(hparams.neurons[i]))

  summary.plot_heatmaps_grid(
      'van_rossum_neuron_heatmaps',
      heatmaps,
      xlabel='Fake trials',
      ylabel='Real trials',
      xticklabels=xticklabels,
      yticklabels=yticklabels,
      titles=titles,
      step=epoch,
      training=False)

  # compute trial-wise van rossum distance histogram
  pool = Pool(hparams.num_processors)
  results = pool.starmap(van_rossum_trial_histogram,
                         [(hparams, filename, i) for i in range(9)])
  pool.close()

  summary.plot_histograms_grid(
      'van_rossum_trial_histograms',
      results,
      xlabel='Distance',
      ylabel='Count',
      titles=['Trial #{:03d}'.format(i) for i in range(len(results))],
      step=epoch,
      training=False)

  # compute neuron-wise van rossum distance histogram for 200 trials
  pool = Pool(hparams.num_processors)
  results = pool.starmap(van_rossum_neuron_histogram,
                         [(hparams, filename, n, 200) for n in hparams.neurons])
  pool.close()

  summary.plot_histograms_grid(
      'van_rossum_neuron_histograms',
      results,
      xlabel='Distance',
      ylabel='Count',
      titles=['Neuron #{:03d}'.format(i) for i in hparams.neurons],
      step=epoch,
      training=False)


def compute_epoch_spike_metrics(hparams, summary, filename, epoch):
  if not h5_helper.contains(filename, 'spikes'):
    deconvolve_from_file(hparams, filename)

  raster_plots(hparams, summary, filename, epoch)

  firing_rate_metrics(hparams, summary, filename, epoch)

  # covariance_metrics(hparams, summary, filename, epoch)

  correlation_coefficient_metrics(hparams, summary, filename, epoch)

  van_rossum_metrics(hparams, summary, filename, epoch)


def main(hparams):
  if not os.path.exists(hparams.output_dir):
    print('{} not found'.format(hparams.output_dir))
    exit()

  utils.load_hparams(hparams)
  info = load_info(hparams)

  if hparams.all_neurons:
    hparams.neurons = list(range(hparams.num_neurons))
  else:
    hparams.neurons = [87, 58, 90, 39, 7, 60, 14, 5, 13]

  summary = Summary(hparams)

  epochs = sorted(list(info.keys()))

  # only compute metrics for the last generated file
  if not hparams.all_epochs:
    epochs = [epochs[-1]]

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
  parser.add_argument('--all_neurons', action='store_true')
  parser.add_argument('--all_epochs', action='store_true')
  parser.add_argument('--dpi', default=120, type=int)
  parser.add_argument('--verbose', default=1, type=int)
  hparams = parser.parse_args()

  warnings.simplefilter(action='ignore', category=UserWarning)
  warnings.simplefilter(action='ignore', category=RuntimeWarning)
  warnings.simplefilter(action='ignore', category=DeprecationWarning)

  main(hparams)
