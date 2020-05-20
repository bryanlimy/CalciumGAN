import os
import pickle
import platform
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from gan.utils import utils
from gan.utils import h5_helper
from dataset.dg.dichot_gauss import DichotGauss
from dataset.dg.optim_dichot_gauss import DGOptimise

import matplotlib

if platform.system() == 'Darwin':
  matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.style.use('seaborn-deep')

import seaborn as sns

tick_size = 20
label_size = 30
legend_size = 14
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


def get_data_statistics(hparams, filename, trial=0):
  ''' Get mean firing rate and correlation of recorded data '''
  spike_trains = h5_helper.get(filename, name='spikes', trial=trial)

  # reshape to (time bins, trial, num neurons)
  spike_trains = np.expand_dims(spike_trains, axis=0)

  dg_optimizer = DGOptimise(spike_trains)

  mean = dg_optimizer.gauss_mean
  corr = dg_optimizer.data_tfix_covariance

  diag_indices = np.triu_indices(len(corr))
  corr = corr[diag_indices]

  return mean, corr


def plot_statistics(filename, dg_means, fake_means, dg_corrs, fake_corrs):
  fig = plt.figure(figsize=(32, 10))
  fig.patch.set_facecolor('white')

  x = list(range(len(dg_means)))

  # plot means
  plt.subplot(1, 2, 1)
  sns.regplot(
      x, dg_means, marker='o', color='dodgerblue', scatter_kws={'alpha': 0.7})
  ax = sns.regplot(
      x, fake_means, marker='x', color='orangered', scatter_kws={'alpha': 0.7})
  ax.set_ylim(
      min(dg_means + fake_means) - 0.05,
      max(dg_means + fake_means) + 0.05)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.set_xlabel('Trials')
  ax.set_ylabel('Mean')
  plt.legend(loc='upper left', labels=['DG', 'CalciumGAN'])

  # plot correlations
  plt.subplot(1, 2, 2)
  sns.regplot(
      x, dg_corrs, marker='o', color='dodgerblue', scatter_kws={'alpha': 0.7})
  ax = sns.regplot(
      x, fake_corrs, marker='x', color='orangered', scatter_kws={'alpha': 0.7})
  ax.set_ylim(
      min(dg_corrs + fake_corrs) - 0.00001,
      max(dg_corrs + fake_corrs) + 0.00001)
  ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.set_xlabel('Trials')
  ax.set_ylabel('Mean correlations')
  plt.legend(loc='upper left', labels=['DG', 'CalciumGAN'])

  plt.tight_layout()
  plt.savefig(filename, dpi=120, format='pdf', transparent=True)
  plt.close()

  print('saved figure to {}'.format(filename))


def main(hparams):
  if not os.path.exists(hparams.output_dir):
    print('{} not found'.format(hparams.output_dir))
    exit()

  utils.load_hparams(hparams)
  info = load_info(hparams)

  epochs = sorted(list(info.keys()))

  hparams.num_trials = h5_helper.get_dataset_length(hparams.validation_cache,
                                                    'spikes')

  dg_means, fake_means = [], []
  dg_corrs, fake_corrs = [], []
  for trial in tqdm(range(hparams.num_trials), desc='Measure statisitcs'):
    dg_mean, dg_corr = get_data_statistics(
        hparams, filename=hparams.validation_cache, trial=trial)
    fake_mean, fake_corr = get_data_statistics(
        hparams, filename=info[epochs[-1]]['filename'], trial=trial)

    dg_means.append(np.mean(dg_mean))
    fake_means.append(np.mean(fake_mean))
    dg_corrs.append(np.mean(dg_corr))
    fake_corrs.append(np.mean(fake_corr))

  plot_statistics('diagrams/dg_statistics.pdf', dg_means, fake_means, dg_corrs,
                  fake_corrs)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', default='runs', type=str)
  hparams = parser.parse_args()
  main(hparams)
