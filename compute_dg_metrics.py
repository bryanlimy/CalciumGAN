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
plt.style.use('seaborn-deep')

import seaborn as sns


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

  return mean, corr


def plot_means(hparams, dg_means, fake_means):
  df = pd.DataFrame({
      'means':
      np.concatenate([dg_means, fake_means]),
      'data': ['DG'] * len(dg_means) + ['Fake'] * len(fake_means),
      'trials':
      list(range(len(dg_means))) + list(range(len(dg_means))),
  })

  g = sns.lmplot(
      x='trials',
      y='means',
      hue='data',
      data=df,
      markers=['o', 'x'],
      palette={
          'DG': "dodgerblue",
          'Fake': 'orangered'
      },
      legend_out=False,
      scatter_kws={'alpha': 0.8},
      line_kws={'alpha': 0.9})
  g.set(ylim=(df['means'].min() - 0.1, df['means'].max() + 0.1))

  plt.gcf().set_size_inches(12, 9)
  axis = plt.gca()
  axis.spines['top'].set_visible(False)
  axis.spines['right'].set_visible(False)
  plt.legend(loc='upper left', labels=['DG', 'CalciumGAN'])
  plt.xlabel('Trials')
  plt.ylabel('Mean')

  plt.tight_layout()
  filename = 'dg_means_lmplot.png'
  plt.savefig(filename, dpi=120)
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

  plot_means(hparams, dg_means, fake_means)

  # print('Trial {:02}/{:02d}'.format(trial + 1, hparams.num_trials))
  # print('dg mean: {:.04f}\tfake mean: {:.04f}'.format(
  #     np.mean(dg_mean), np.mean(fake_mean)))
  # print('corr mean: {:.04f}\tcorr mean: {:.04f}'.format(
  #     np.mean(dg_corr), np.mean(fake_corr)))
  # print('mean all close: {}'.format(
  #     np.allclose(dg_mean, fake_mean, rtol=0.2, atol=0.8)))
  # print('corr all close: {}\n'.format(
  #     np.allclose(dg_corr, fake_corr, rtol=0.2, atol=0.8)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', default='runs', type=str)
  hparams = parser.parse_args()
  main(hparams)
