import os
import pickle
import random
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


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)


def load_info(hparams):
  filename = os.path.join(hparams.generated_dir, 'info.pkl')
  with open(filename, 'rb') as file:
    info = pickle.load(file)
  return info


def deconvolve_neuron(hparams, filename, neuron):
  signals = h5_helper.get(filename, name='signals', neuron=neuron)
  signals = utils.set_array_format(signals, data_format='NW', hparams=hparams)
  return spike_helper.deconvolve_signals(signals, threshold=0.5)


def deconvolve_from_file(hparams, filename, return_spikes=False):
  if hparams.verbose:
    print('\tDeconvolve {}'.format(filename))

  pool = Pool(hparams.num_processors)
  fake_spikes = pool.starmap(
      deconvolve_neuron,
      [(hparams, filename, n) for n in range(hparams.num_neurons)])
  pool.close()

  fake_spikes = utils.set_array_format(
      np.array(fake_spikes, dtype=np.int8), data_format='NWC', hparams=hparams)

  h5_helper.write(filename, {'spikes': fake_spikes})

  if return_spikes:
    return fake_spikes


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


def kl_divergence(p, q):
  # replace entries with 0 probability with 1e-10
  p = np.where(p == 0, 1e-10, p)
  q = np.where(q == 0, 1e-10, q)
  return np.sum(p * np.log(p / q))


def pairs_kl_divergence(pairs):
  kl = np.zeros((len(pairs),), dtype=np.float32)
  for i in range(len(pairs)):
    real, fake = pairs[i]

    df = pd.DataFrame({
        'data': np.concatenate([real, fake]),
        'is_real': [True] * len(real) + [False] * len(fake)
    })

    num_bins = 30
    df['bins'] = pd.cut(df.data, bins=num_bins, labels=np.arange(num_bins))

    real_pdf = np.array([
        len(df[(df.bins == i) & (df.is_real == True)]) for i in range(num_bins)
    ],
                        dtype=np.float32) / len(real)
    fake_pdf = np.array([
        len(df[(df.bins == i) & (df.is_real == False)]) for i in range(num_bins)
    ],
                        dtype=np.float32) / len(fake)

    kl[i] = kl_divergence(real_pdf, fake_pdf)
  return kl


def plot_signals(hparams, summary, filename, epoch):
  trial = random.randint(0, hparams.num_samples)

  if hparams.verbose:
    print('\tPlotting traces for trial #{}'.format(trial))

  real_signals = h5_helper.get(
      hparams.validation_cache, name='signals', trial=trial)
  real_spikes = h5_helper.get(
      hparams.validation_cache, name='spikes', trial=trial)

  real_signals = utils.set_array_format(
      real_signals, data_format='CW', hparams=hparams)
  real_spikes = utils.set_array_format(
      real_spikes, data_format='CW', hparams=hparams)

  fake_signals = h5_helper.get(filename, name='signals', trial=trial)
  fake_spikes = h5_helper.get(filename, name='spikes', trial=trial)

  fake_signals = utils.set_array_format(
      fake_signals, data_format='CW', hparams=hparams)
  fake_spikes = utils.set_array_format(
      fake_spikes, data_format='CW', hparams=hparams)

  # get the y axis range for each neuron
  assert real_signals.shape == fake_signals.shape
  ylims = []
  for i in range(len(real_signals)):
    ylims.append([
        np.min([np.min(real_signals[i]),
                np.min(fake_signals[i])]),
        np.max([np.max(real_signals[i]),
                np.max(fake_signals[i])])
    ])

  summary.plot_traces(
      'real',
      real_signals,
      real_spikes,
      indexes=hparams.neurons[:6],
      ylims=ylims,
      step=epoch)

  summary.plot_traces(
      'fake',
      fake_signals,
      fake_spikes,
      indexes=hparams.neurons[:6],
      ylims=ylims,
      step=epoch)


def raster_plots(hparams, summary, filename, epoch, trial=1):
  if hparams.verbose:
    print('\tPlotting raster plot for trial #{}'.format(trial))

  real_spikes = h5_helper.get(
      hparams.validation_cache, name='spikes', trial=trial)
  real_spikes = utils.set_array_format(real_spikes, 'CW', hparams)
  fake_spikes = h5_helper.get(filename, name='spikes', trial=trial)
  fake_spikes = utils.set_array_format(fake_spikes, 'CW', hparams)

  summary.raster_plot(
      'raster_plot_trial',
      real_spikes=real_spikes,
      fake_spikes=fake_spikes,
      xlabel='Time (ms)',
      ylabel='Neuron',
      title='Trial #{}'.format(trial),
      step=epoch)


def firing_rate(hparams, filename, neuron, num_trials=200):
  if hparams.verbose == 2:
    print('\tComputing firing rate for neuron #{}'.format(neuron))

  real_spikes = get_neo_trains(
      hparams,
      hparams.validation_cache,
      neuron=neuron,
      data_format='NW',
      num_trials=num_trials)
  fake_spikes = get_neo_trains(
      hparams,
      filename,
      neuron=neuron,
      data_format='NW',
      num_trials=num_trials,
  )

  real_firing_rate = spike_metrics.mean_firing_rate(real_spikes)
  fake_firing_rate = spike_metrics.mean_firing_rate(fake_spikes)

  return (real_firing_rate, fake_firing_rate)


def firing_rate_metrics(hparams, summary, filename, epoch):
  if hparams.verbose:
    print('\tComputing firing rate')

  pool = Pool(hparams.num_processors)
  firing_rate_pairs = pool.starmap(
      firing_rate, [(hparams, filename, n, min(hparams.num_samples, 1000))
                    for n in range(hparams.num_neurons)])
  pool.close()

  summary.plot_histograms_grid(
      'firing_rate_histograms',
      data=[firing_rate_pairs[n] for n in hparams.neurons],
      xlabel='Hz',
      ylabel='Count',
      titles=['Neuron #{:03d}'.format(n) for n in hparams.neurons],
      step=epoch)

  kl_divergence = pairs_kl_divergence(firing_rate_pairs)
  summary.plot_distribution(
      'firing_rate_kl',
      data=kl_divergence,
      xlabel='KL divergence',
      ylabel='Count',
      title='Firing rate KL divergence',
      step=epoch)

  if hparams.verbose:
    print(
        '\tmin: {:.04f}, max: {:.04f}, mean: {:.04f}, num below 1.5: {}'.format(
            np.min(kl_divergence), np.max(kl_divergence),
            np.mean(kl_divergence), np.count_nonzero(kl_divergence < 1.5)))


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

  # compute neuron-wise covariance
  pool = Pool(hparams.num_processors)
  results = pool.starmap(neuron_covariance,
                         [(hparams, filename, n, 200) for n in hparams.neurons])
  pool.close()

  summary.scalar(
      'spike_metrics/neuron_covariance', np.mean(results), step=epoch)


def correlation_coefficient(hparams, filename, trial):
  if hparams.verbose == 2:
    print('\t\tComputing correlation coefficient for sample #{}'.format(trial))

  diag_indices = np.triu_indices(hparams.num_neurons, k=1)

  real_spikes = get_neo_trains(
      hparams, hparams.validation_cache, trial=trial, data_format='CW')
  real_corrcoef = spike_metrics.correlation_coefficients(real_spikes, None)
  real_corrcoef = utils.remove_nan(real_corrcoef[diag_indices])

  fake_spikes = get_neo_trains(hparams, filename, trial=trial, data_format='CW')
  fake_corrcoef = spike_metrics.correlation_coefficients(fake_spikes, None)
  fake_corrcoef = utils.remove_nan(fake_corrcoef[diag_indices])

  return (real_corrcoef, fake_corrcoef)


def correlation_coefficient_metrics(hparams, summary, filename, epoch):
  if hparams.verbose:
    print('\tComputing correlation coefficient')

  pool = Pool(hparams.num_processors)
  correlations = pool.starmap(
      correlation_coefficient,
      [(hparams, filename, i) for i in range(min(hparams.num_samples, 1000))])
  pool.close()

  summary.plot_histograms_grid(
      'correlation_histogram',
      data=[correlations[i] for i in hparams.trials],
      xlabel='Correlation',
      ylabel='Count',
      titles=['Trial #{:03d}'.format(i) for i in hparams.trials],
      step=epoch)

  kl_divergence = pairs_kl_divergence(correlations)
  summary.plot_distribution(
      'correlation_kl',
      data=kl_divergence,
      xlabel='KL divergence',
      ylabel='Count',
      title='Correlation coefficient KL divergence',
      step=epoch)

  if hparams.verbose:
    print(
        '\tmin: {:.04f}, max: {:.04f}, mean: {:.04f}, num below 1.5: {}'.format(
            np.min(kl_divergence), np.max(kl_divergence),
            np.mean(kl_divergence), np.count_nonzero(kl_divergence < 1.5)))


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


def neuron_van_rossum(hparams, filename, neuron, num_trials=50):
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


def trial_van_rossum(hparams, filename, trial):
  ''' compute van rossum distance for a given trial '''
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


def van_rossum_metrics(hparams, summary, filename, epoch):
  if hparams.verbose:
    print('\tComputing van-rossum distance')

  # compute van-Rossum distance heatmap
  pool = Pool(hparams.num_processors)
  results = pool.starmap(neuron_van_rossum,
                         [(hparams, filename, n, 50) for n in hparams.neurons])
  pool.close()

  heatmaps, xticklabels, yticklabels, titles = [], [], [], []
  for i in range(len(results)):
    heatmaps.append(results[i]['heatmap'])
    xticklabels.append(results[i]['xticklabels'])
    yticklabels.append(results[i]['yticklabels'])
    titles.append('Neuron #{:03d}'.format(hparams.neurons[i]))

  summary.plot_heatmaps_grid(
      'van_rossum_heatmaps',
      matrix=heatmaps,
      xlabel='Fake trials',
      ylabel='Real trials',
      xticklabels=xticklabels,
      yticklabels=yticklabels,
      titles=titles,
      step=epoch)

  # compute van rossum distance KL divergence
  pool = Pool(hparams.num_processors)
  van_rossum_pairs = pool.starmap(
      trial_van_rossum,
      [(hparams, filename, i) for i in range(min(hparams.num_samples, 1000))])
  pool.close()

  kl_divergence = pairs_kl_divergence(van_rossum_pairs)
  summary.plot_distribution(
      'van_rossum_kl_divergence',
      data=kl_divergence,
      xlabel='KL divergence',
      ylabel='Count',
      title='van-Rossum distance KL divergence',
      step=epoch)

  if hparams.verbose:
    print(
        '\tmin: {:.04f}, max: {:.04f}, mean: {:.04f}, num below 1.5: {}'.format(
            np.min(kl_divergence), np.max(kl_divergence),
            np.mean(kl_divergence), np.count_nonzero(kl_divergence < 1.5)))


def compute_epoch_spike_metrics(hparams, summary, filename, epoch):
  if not h5_helper.contains(filename, 'spikes'):
    deconvolve_from_file(hparams, filename)

  plot_signals(hparams, summary, filename, epoch)

  raster_plots(hparams, summary, filename, epoch)

  firing_rate_metrics(hparams, summary, filename, epoch)

  # covariance_metrics(hparams, summary, filename, epoch)

  correlation_coefficient_metrics(hparams, summary, filename, epoch)

  van_rossum_metrics(hparams, summary, filename, epoch)


def main(hparams):
  if not os.path.exists(hparams.output_dir):
    print('{} not found'.format(hparams.output_dir))
    exit()

  set_seed(hparams.seed)

  utils.load_hparams(hparams)
  info = load_info(hparams)

  hparams.num_samples = h5_helper.get_dataset_length(hparams.validation_cache,
                                                     'signals')

  # randomly select neurons and trials to plot
  hparams.neurons = list(
      range(hparams.num_neurons
           ) if hparams.num_neuron_plots >= hparams.num_neurons else np.random.
      choice(hparams.num_neurons, hparams.num_neuron_plots))
  hparams.trials = list(
      np.random.choice(hparams.num_samples, hparams.num_trial_plots))

  summary = Summary(hparams, spike_metrics=True)

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

    summary.scalar('elapse/spike_metrics', end - start, step=epoch)

    if hparams.verbose:
      print('{} took {:.02f} mins'.format(info[epoch]['filename'],
                                          (end - start) / 60))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', default='runs')
  parser.add_argument('--num_processors', default=6, type=int)
  parser.add_argument('--all_epochs', action='store_true')
  parser.add_argument('--num_neuron_plots', default=6, type=int)
  parser.add_argument('--num_trial_plots', default=6, type=int)
  parser.add_argument('--dpi', default=120, type=int)
  parser.add_argument('--verbose', default=1, type=int)
  parser.add_argument('--seed', default=12, type=int)
  hparams = parser.parse_args()

  warnings.simplefilter(action='ignore', category=UserWarning)
  warnings.simplefilter(action='ignore', category=RuntimeWarning)
  warnings.simplefilter(action='ignore', category=DeprecationWarning)

  main(hparams)
