import os
import pickle
import warnings
import platform
import argparse
import numpy as np
from tqdm import tqdm

import analysis
from calciumgan.utils import utils
from calciumgan.utils import spike_metrics

import matplotlib

if platform.system() == 'Darwin':
  matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

import seaborn as sns

tick_size = 11
legend_size = 11
label_size = 13
plt.rc('xtick', labelsize=tick_size)
plt.rc('ytick', labelsize=tick_size)
plt.rc('axes', titlesize=label_size)
plt.rc('axes', labelsize=label_size)
plt.rc('legend', fontsize=legend_size)


def load_info(hparams):
  filename = os.path.join(hparams.generated_dir, 'info.pkl')
  with open(filename, 'rb') as file:
    info = pickle.load(file)
  return info


def get_data_statistics(hparams, filename):
  ''' Get mean firing rate and correlation of recorded data '''
  firing_rates = np.zeros(
      shape=(hparams.num_neurons, hparams.num_trials), dtype=np.float32)
  covariances = np.zeros(
      shape=(hparams.num_neurons * (hparams.num_neurons + 1) // 2,
             hparams.num_trials),
      dtype=np.float32)

  for i in tqdm(range(hparams.num_trials), desc="Trial"):
    spike_trains = analysis.get_neo_trains(
        hparams, filename, trial=i, data_format='CW')
    firing_rates[:, i] = spike_metrics.mean_firing_rate(spike_trains)
    covariance = spike_metrics.covariance(spikes1=spike_trains, spikes2=None)
    indices = np.triu_indices(len(covariance))
    covariance = np.nan_to_num(covariance[indices])
    covariances[:, i] = covariance

  return firing_rates, covariances


def plot_firing_rate(hparams, filename, real, fake):
  assert real.shape == fake.shape

  # sort firing rate by the mean of num_trials trials
  neuron_order = np.argsort(np.mean(real, axis=-1))
  real = real[neuron_order].flatten('F')
  fake = fake[neuron_order].flatten('F')
  x = list(range(len(neuron_order)))

  fig = plt.figure(figsize=(5, 4))
  fig.patch.set_facecolor('white')

  scatter_kws = {'alpha': 0.6}
  sns.regplot(
      x=x * hparams.num_trials,
      y=real,
      marker='o',
      fit_reg=False,
      color='dodgerblue',
      scatter_kws=scatter_kws)
  ax = sns.regplot(
      x=x * hparams.num_trials,
      y=fake,
      marker='x',
      fit_reg=False,
      color='orangered',
      scatter_kws=scatter_kws)

  plt.xticks(ticks=list(range(0, len(x), 5)), labels=neuron_order, rotation=90)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.set_xlabel('Neuron')
  ax.set_ylabel('Firing rate')
  plt.legend(loc='upper left', labels=['DG', 'CalciumGAN'], frameon=False)

  plt.tight_layout()
  plt.savefig(filename, dpi=120, format='pdf', transparent=True)
  plt.close()

  print('saved firing rate figure to {}'.format(filename))


def plot_covariance(hparams, filename, real, fake):
  assert real.shape == fake.shape

  # sort covariance by the mean of num_trials trials
  pair_order = np.argsort(np.mean(real, axis=-1))
  # plot every 10th pair so that the graph won't be too clustered
  pair_order = pair_order[::10]
  real = real[pair_order].flatten('F')
  fake = fake[pair_order].flatten('F')
  x = list(range(len(pair_order)))

  fig = plt.figure(figsize=(5, 4))
  fig.patch.set_facecolor('white')

  scatter_kws = {'alpha': 0.6}
  sns.regplot(
      x=x * hparams.num_trials,
      y=real,
      fit_reg=False,
      marker='o',
      color='dodgerblue',
      scatter_kws=scatter_kws)
  ax = sns.regplot(
      x=x * hparams.num_trials,
      y=fake,
      fit_reg=False,
      marker='x',
      color='orangered',
      scatter_kws=scatter_kws)

  plt.xticks(ticks=list(range(0, len(x), 20)), labels=pair_order, rotation=90)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.set_xlabel('Neuron Pair')
  ax.set_ylabel('Covariance')

  plt.tight_layout()
  plt.savefig(filename, dpi=120, format='pdf', transparent=True)
  plt.close()

  print('saved covariance figure to {}'.format(filename))


def percentage_error(y_true, y_pred):
  error = np.empty(y_true.shape)
  for j in range(y_true.shape[0]):
    if y_true[j] != 0.0:
      error[j] = (y_true[j] - y_pred[j]) / y_true[j]
    else:
      error[j] = y_pred[j] / np.mean(y_true)
  return error


def mean_absolute_percentage_error(y_true, y_pred):
  errors = np.zeros(shape=y_true.shape, dtype=np.float32)
  for i in range(errors.shape[1]):
    errors[..., i] = percentage_error(y_true[..., i], y_pred[..., i])
  mape = np.mean(np.abs(errors), axis=0)
  mape = np.mean(mape, axis=0)
  return mape * 100


def main(hparams):
  if not os.path.exists(hparams.output_dir):
    print('{} not found'.format(hparams.output_dir))
    exit()

  utils.load_hparams(hparams)
  info = load_info(hparams)

  epochs = sorted(list(info.keys()))

  real_firing_rate, real_covariance = get_data_statistics(
      hparams, filename=hparams.validation_cache)
  fake_firing_rate, fake_covariance = get_data_statistics(
      hparams, filename=info[epochs[-1]]['filename'])

  if hparams.save_plots:
    plot_firing_rate(
        hparams,
        filename=os.path.join(hparams.output_dir, 'dg_firing_rate.pdf'),
        real=real_firing_rate,
        fake=fake_firing_rate)
    plot_covariance(
        hparams,
        filename=os.path.join(hparams.output_dir, 'dg_covariance.pdf'),
        real=real_covariance,
        fake=fake_covariance)

  print('\nmean firing rate\n\tMAE\t{:.02f}\n\tRMSE\t{:.02f}\n\tMAPE\t{:.02f}%'.
        format(
            np.mean(np.abs(real_firing_rate - fake_firing_rate)),
            np.sqrt(np.mean(np.square(real_firing_rate - fake_firing_rate))),
            mean_absolute_percentage_error(real_firing_rate, fake_firing_rate)))

  print('\ncovariance\n\tMAE\t{:.02f}\n\tMSE\t{:.02f}\n\tMAPE\t{:.02f}%'.format(
      np.mean(np.abs(real_covariance - fake_covariance)),
      np.mean(np.square(real_covariance - fake_covariance)),
      mean_absolute_percentage_error(real_covariance, fake_covariance)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', default='runs', type=str)
  parser.add_argument('--num_trials', default=5, type=int)
  parser.add_argument('--save_plots', action='store_true')
  hparams = parser.parse_args()

  warnings.simplefilter(action='ignore', category=UserWarning)
  warnings.simplefilter(action='ignore', category=RuntimeWarning)
  warnings.simplefilter(action='ignore', category=DeprecationWarning)

  main(hparams)
