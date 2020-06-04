import os
import pickle
import warnings
import platform
import argparse
import numpy as np
from tqdm import tqdm

import compute_metrics
from gan.utils import utils
from gan.utils import spike_metrics

import matplotlib

if platform.system() == 'Darwin':
  matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

import seaborn as sns

tick_size = 24
label_size = 35
legend_size = 25
plt.rc('xtick', labelsize=tick_size)
plt.rc('ytick', labelsize=tick_size)
plt.rc('axes', titlesize=label_size)
plt.rc('axes', labelsize=label_size)
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
    spike_trains = compute_metrics.get_neo_trains(
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

  fig = plt.figure(figsize=(16, 10))
  fig.patch.set_facecolor('white')

  sns.regplot(
      x=x * hparams.num_trials,
      y=real,
      marker='o',
      fit_reg=False,
      color='dodgerblue',
      scatter_kws={'alpha': 0.7})
  ax = sns.regplot(
      x=x * hparams.num_trials,
      y=fake,
      marker='x',
      fit_reg=False,
      color='orangered',
      scatter_kws={'alpha': 0.7})

  plt.xticks(ticks=list(range(0, len(x), 3)), labels=neuron_order, rotation=90)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.set_xlabel('Neuron')
  ax.set_ylabel('Firing rate (Hz)')
  plt.legend(loc='upper left', labels=['DG', 'CalciumGAN'])

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

  fig = plt.figure(figsize=(16, 10))
  fig.patch.set_facecolor('white')

  sns.regplot(
      x=x * hparams.num_trials,
      y=real,
      fit_reg=False,
      marker='o',
      color='dodgerblue',
      scatter_kws={'alpha': 0.7})
  ax = sns.regplot(
      x=x * hparams.num_trials,
      y=fake,
      fit_reg=False,
      marker='x',
      color='orangered',
      scatter_kws={'alpha': 0.7})

  plt.xticks(ticks=list(range(0, len(x), 12)), labels=pair_order, rotation=90)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.set_xlabel('Neuron Pair')
  ax.set_ylabel('Covariance')
  plt.legend(loc='upper left', labels=['DG', 'CalciumGAN'])

  plt.tight_layout()
  plt.savefig(filename, dpi=120, format='pdf', transparent=True)
  plt.close()

  print('saved covariance figure to {}'.format(filename))


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

  plot_firing_rate(
      hparams,
      filename='diagrams/dg_firing_rate.pdf',
      real=real_firing_rate,
      fake=fake_firing_rate)
  plot_covariance(
      hparams,
      filename='diagrams/dg_covariance.pdf',
      real=real_covariance,
      fake=fake_covariance)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', default='runs', type=str)
  parser.add_argument('--num_trials', default=5, type=int)
  hparams = parser.parse_args()

  warnings.simplefilter(action='ignore', category=UserWarning)
  warnings.simplefilter(action='ignore', category=RuntimeWarning)
  warnings.simplefilter(action='ignore', category=DeprecationWarning)

  main(hparams)
