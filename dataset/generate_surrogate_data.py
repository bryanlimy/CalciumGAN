import os
import shutil
import pickle
import elephant
import argparse
import numpy as np
from tqdm import tqdm
import quantities as pq
from neo.core import SpikeTrain

np.random.seed(1234)

from dg.dichot_gauss import DichotGauss
from dg.optim_dichot_gauss import DGOptimise


def generate_dg_spikes(hparams, mean, corr):
  spike_trains = np.zeros(
      (hparams.num_trials, hparams.num_neurons, hparams.sequence_length),
      dtype=np.float32)

  dg = DichotGauss(hparams.num_neurons, mean=mean, corr=corr, make_pd=True)

  for i in tqdm(range(hparams.num_trials), desc='Sampling'):
    spikes = dg.sample(repeats=hparams.sequence_length)
    # reshape to (num_neurons, duration)
    spikes = np.squeeze(spikes, axis=0)
    spikes = np.transpose(spikes, axes=[1, 0])
    spike_trains[i] = spikes
  return spike_trains


def spikes_to_signals(hparams, spikes, g=[.95], sn=.3, b=0):
  ''' Convert spike trains to calcium signals 
  Code extracted from https://github.com/j-friedrich/OASIS/blob/e62063cfd8bc0f06625aebd3ea3e09133665b409/oasis/functions.py#L17
  '''
  signals = np.zeros(spikes.shape, dtype=np.float32)
  for i in tqdm(range(spikes.shape[0]), desc='Transformation'):
    spike = spikes[i].astype(np.float32)
    for j in range(2, spike.shape[1]):
      if len(g) == 2:
        spike[:, j] += g[0] * spike[:, j - 1] + g[1] * spike[:, j - 2]
      else:
        spike[:, j] += g[0] * spike[:, j - 1]
    signals[i] = b + spike + sn * np.random.randn(spike.shape[0],
                                                  spike.shape[1])
  return signals


def main(hparams):
  if os.path.exists(hparams.output_dir):
    shutil.rmtree(hparams.output_dir)
  os.makedirs(hparams.output_dir)

  hparams.num_neurons = 2
  hparams.sequence_length = 12
  mean = np.array([[.4, .3]], dtype=np.float32)
  covariance = np.eye(hparams.num_neurons)
  covariance[0, 1], covariance[1, 0] = 0.3, 0.3

  # generate surrogate dataset
  surrogate = generate_dg_spikes(hparams, mean, covariance)
  print('save surrogate dataset to {}'.format(hparams.surrogate_path))
  with open(hparams.surrogate_path, 'wb') as file:
    pickle.dump({'spikes': surrogate}, file)

  # generate ground truth dataset
  ground_truth = generate_dg_spikes(hparams, mean, covariance)
  with open(hparams.ground_truth_path, 'wb') as file:
    pickle.dump({'spikes': ground_truth}, file)

  # select subset for training
  indices = np.random.choice(len(ground_truth), size=hparams.training_size)
  training_spikes = ground_truth[indices]
  training_signals = spikes_to_signals(hparams, training_spikes)
  with open(hparams.training_path, 'wb') as file:
    pickle.dump({'spikes': training_spikes, 'signals': training_signals}, file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', default='surrogate', type=str)
  parser.add_argument('--num_trials', default=2 * 10**6, type=int)
  parser.add_argument('--training_size', default=9216, type=int)
  hparams = parser.parse_args()

  hparams.surrogate_path = os.path.join(hparams.output_dir, 'surrogate.pkl')
  hparams.ground_truth_path = os.path.join(hparams.output_dir,
                                           'ground_truth.pkl')
  hparams.training_path = os.path.join(hparams.output_dir, 'training.pkl')

  main(hparams)
