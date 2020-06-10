import os
import pickle
import warnings
import platform
import argparse
import numpy as np
from tqdm import tqdm

from gan.utils import utils
from gan.utils import spike_helper
from dataset.generate_surrogate_data import spikes_to_signals

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


def deconvolution(signals):
  spikes = np.zeros(signals.shape, dtype=np.int32)
  for i in tqdm(range(len(signals)), desc='Deconvolution'):
    spikes[i] = spike_helper.deconvolve_signals(signals[i])
  return spikes


def spikes_conversion(data):
  data['signals'] = spikes_to_signals(data['spikes'])
  data['spikes'] = deconvolution(data['signals'])
  return data


def get_pickle_data(filename):
  print('reading from {}...'.format(filename))
  with open(filename, 'rb') as file:
    data = pickle.load(file)

  data = spikes_conversion(data)
  data['unique'], data['count'] = np.unique(
      data['spikes'], return_counts=True, axis=0)

  return data


def get_generate_data(hparams):
  filename = os.path.join(hparams.output_dir, 'generated.pkl')
  if not os.path.exists(filename):
    print('generated pickle file not found in {}'.format(hparams.output_dir))
    exit()

  print('reading from {}...'.format(filename))

  with open(filename, 'rb') as file:
    data = pickle.load(file)

  data['signals'] = utils.set_array_format(
      data['signals'], data_format='NCW', hparams=hparams)
  data['spikes'] = deconvolution(data['signals'])
  data['unique'], data['count'] = np.unique(
      data['spikes'], return_counts=True, axis=0)

  return data


def get_probability(sequence):
  shape = sequence.shape
  flattened = np.reshape(sequence, newshape=(shape[0], shape[1] * shape[2]))
  unique, indices, counts = np.unique(
      flattened, return_inverse=True, return_counts=True, axis=0)
  prob = counts[indices] / shape[0]
  return prob


def get_probabilities(joint_unique_samples, data):
  unique_samples, unique_count = data['unique'], data['count']

  counts = np.zeros((joint_unique_samples.shape[0],), dtype=np.int32)

  for i in tqdm(range(len(joint_unique_samples)), desc='Count unique'):
    for j in range(len(unique_count)):
      if np.array_equal(joint_unique_samples[i], unique_samples[j]):
        counts[i] = unique_count[j]
        break

  probabilities = counts / len(data['spikes']) + 1e-5
  return probabilities


def main(hparams):
  if not os.path.exists(hparams.output_dir):
    print('{} not found'.format(hparams.output_dir))
    exit()

  utils.load_hparams(hparams)

  ground_truth_path = os.path.join(hparams.input_dir, 'ground_truth.pkl')
  ground_truth_data = get_pickle_data(ground_truth_path)

  surrogate_path = os.path.join(hparams.input_dir, 'surrogate.pkl')
  surrogate_data = get_pickle_data(surrogate_path)

  generated_data = get_generate_data(hparams)

  joint_unique_samples = np.unique(
      np.concatenate([
          ground_truth_data['unique'], surrogate_data['unique'],
          generated_data['unique']
      ],
                     axis=0),
      axis=0)

  ground_truth_prob = get_probabilities(joint_unique_samples, ground_truth_data)
  surrogate_prob = get_probabilities(joint_unique_samples, surrogate_data)
  generated_prob = get_probabilities(joint_unique_samples, generated_data)

  filename = 'diagrams/numerical_probabilities.pdf'

  ground_truth_prob = np.log10(ground_truth_prob)
  surrogate_prob = np.log10(surrogate_prob)
  generated_prob = np.log10(generated_prob)

  print('min {:.04f}\tmax {:.04f}'.format(
      min(
          np.min(ground_truth_prob), np.min(surrogate_prob),
          np.min(generated_prob)),
      max(
          np.max(surrogate_prob), np.max(ground_truth_prob),
          np.max(generated_prob))))

  clip = (min(
      np.min(surrogate_prob), np.min(ground_truth_prob),
      np.min(generated_prob)),
          max(
              np.max(surrogate_prob), np.max(ground_truth_prob),
              np.max(generated_prob)))

  # clip = (0.0, 0.01)

  plt.figure(figsize=(8, 8))
  ax = sns.kdeplot(
      data=surrogate_prob,
      data2=ground_truth_prob,
      shade=True,
      shade_lowest=False,
      cmap="Blues",
      clip=clip)
  ax = sns.kdeplot(
      data=generated_prob,
      data2=ground_truth_prob,
      shade=True,
      shade_lowest=False,
      cmap="Reds",
      clip=clip)
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
