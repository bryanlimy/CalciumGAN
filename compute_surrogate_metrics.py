import os
import pickle
import warnings
import platform
import argparse
import numpy as np
from tqdm import tqdm

from gan.utils import utils
from gan.utils import spike_helper

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


def get_pickle_data(filename, name='spikes'):
  with open(filename, 'rb') as file:
    data = pickle.load(file)
  return data[name]


def get_generate_data(hparams):
  filename = os.path.join(hparams.output_dir, 'generated.pkl')
  if not os.path.exists(filename):
    print('generated pickle file not found in {}'.format(hparams.output_dir))
    exit()
  with open(filename, 'rb') as file:
    data = pickle.load(file)
  if 'spikes' in data:
    spikes = data['spikes']
  else:
    signals = utils.set_array_format(
        data['signals'], data_format='NCW', hparams=hparams)
    spikes = np.zeros(signals.shape, dtype=np.float32)
    for i in tqdm(range(len(signals)), desc='Deconvolution'):
      spikes[i] = spike_helper.deconvolve_signals(signals[i], threshold=0.5)
    with open(filename, 'wb') as file:
      pickle.dump({'signals': signals, 'spikes': spikes}, file)
  return utils.set_array_format(spikes, data_format='NCW', hparams=hparams)


def get_probability(sequence, filename=None, recompute=False):
  if not recompute and filename is not None:
    with open(filename, 'rb') as file:
      data = pickle.load(file)
    if 'probabilities' in data:
      return data['probabilities']

  shape = sequence.shape
  flattened = np.reshape(sequence, newshape=(shape[0], shape[1] * shape[2]))
  unique, indices, counts = np.unique(
      flattened, return_inverse=True, return_counts=True, axis=0)
  prob = counts[indices] / shape[0]
  if filename is not None:
    with open(filename, 'rb') as file:
      data = pickle.load(file)
    data['probabilities'] = prob
    with open(filename, 'wb') as file:
      pickle.dump(data, file)

  return prob


def main(hparams):
  if not os.path.exists(hparams.output_dir):
    print('{} not found'.format(hparams.output_dir))
    exit()

  utils.load_hparams(hparams)

  print('get probability for the ground truth dataset')
  ground_truth_path = os.path.join(hparams.input_dir, 'ground_truth.pkl')
  ground_truth_data = get_pickle_data(ground_truth_path)
  ground_truth_prob = get_probability(
      ground_truth_data,
      filename=ground_truth_path,
      recompute=hparams.recompute)

  print('get probability for the surrogate dataset')
  surrogate_path = os.path.join(hparams.input_dir, 'surrogate.pkl')
  surrogate_data = get_pickle_data(surrogate_path)
  surrogate_prob = get_probability(
      surrogate_data, filename=surrogate_path, recompute=hparams.recompute)

  print('get probability for the generated dataset')
  generated_data = get_generate_data(hparams)
  generated_prob = get_probability(
      generated_data,
      filename=os.path.join(hparams.output_dir, 'generated.pkl'),
      recompute=hparams.recompute)

  def print_min_max(array):
    print('min: {:.04f}, max: {:.04f}, mean: {:.04f}'.format(
        np.min(array), np.max(array), np.mean(array)))

  filename = 'diagrams/numerical_probabilities.pdf'

  ground_truth_prob = np.log10(ground_truth_prob)
  surrogate_prob = np.log10(surrogate_prob)
  generated_prob = np.log10(generated_prob)

  ground_truth_prob = np.sort(ground_truth_prob)
  surrogate_prob = np.sort(surrogate_prob)
  generated_prob = np.sort(generated_prob)

  # plt.figure(figsize=(8, 8))
  sns.kdeplot(
      data=surrogate_prob,
      data2=ground_truth_prob,
      shade=True,
      shade_lowest=False,
      cmap="Blues")
  ax = sns.kdeplot(
      data=generated_prob,
      data2=ground_truth_prob,
      shade=True,
      shade_lowest=False,
      cmap="Reds")
  ax.set_xlabel('log probabilities of surrogate and generated data')
  ax.set_ylabel('log probabilities of ground truth data')
  # ax.legend(
  #     labels=['surrogate', 'generated'], ncol=2, frameon=True, loc='top right')
  plt.tight_layout()
  plt.savefig(filename, dpi=120, format='pdf', transparent=True)
  plt.close()

  print('save numerical probabilities figure to {}'.format(filename))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', default='runs', type=str)
  parser.add_argument('--num_processors', default=4, type=int)
  parser.add_argument('--recompute', action='store_true')
  hparams = parser.parse_args()

  warnings.simplefilter(action='ignore', category=UserWarning)
  warnings.simplefilter(action='ignore', category=RuntimeWarning)
  warnings.simplefilter(action='ignore', category=DeprecationWarning)

  main(hparams)
