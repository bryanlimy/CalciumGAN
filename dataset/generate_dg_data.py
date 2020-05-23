import os
import pickle
import elephant
import argparse
import numpy as np
import quantities as pq
from neo.core import SpikeTrain

np.random.seed(1234)

from dg.dichot_gauss import DichotGauss
from dg.optim_dichot_gauss import DGOptimise


def trains_to_neo(trains):
  ''' convert array of spike trains to list of  Neo SpikeTrains in sec scale '''
  t_stop = trains.shape[-1] * pq.ms
  return [
      SpikeTrain(
          np.nonzero(trains[i])[0] * pq.ms,
          units=pq.s,
          t_stop=t_stop,
          dtype=np.float32) for i in range(len(trains))
  ]


def mean_firing_rate(spikes):
  ''' get mean firing rate of spikes in Hz'''
  result = [
      elephant.statistics.mean_firing_rate(spikes[i])
      for i in range(len(spikes))
  ]
  return np.array(result, dtype=np.float32)


def correlation_coefficients(spike_trains, binsize=100 * pq.ms):
  binned = elephant.conversion.BinnedSpikeTrain(spike_trains, binsize=binsize)
  result = elephant.spike_train_correlation.corrcoef(binned, fast=False)
  return np.nan_to_num(result)


def get_recorded_data_statistics(hparams):
  ''' Get mean firing rate and correlation of recorded data '''
  if not os.path.exists(hparams.input):
    print('Input {} does not exists'.format(hparams.input))
    exit()

  with open(hparams.input, 'rb') as file:
    data = pickle.load(file)

  spike_trains = np.array(data['oasis'], dtype=np.float32)[2:]
  hparams.num_neurons = spike_trains.shape[0]
  hparams.duration = spike_trains.shape[1]

  # reshape to (time bins, trial, num neurons)
  spike_trains = np.transpose(spike_trains, axes=(1, 0))
  spike_trains = np.expand_dims(spike_trains, axis=0)

  dg_optimizer = DGOptimise(spike_trains)

  mean = dg_optimizer.gauss_mean
  covariance = dg_optimizer.data_tfix_covariance

  return mean, covariance


def generate_dg_spikes(hparams, mean, corr):
  dg = DichotGauss(hparams.num_neurons, mean=mean, corr=corr, make_pd=True)
  spike_trains = dg.sample(repeats=hparams.duration)

  # reshape to (num_neurons, duration)
  spike_trains = np.squeeze(spike_trains, axis=0)
  spike_trains = np.transpose(spike_trains, axes=[1, 0])

  return spike_trains.astype(np.float32)


def spikes_to_signals(hparams, spike_trains, g=[.95], sn=.3, b=0):
  ''' Convert spike trains to calcium signals 
  Code extracted from https://github.com/j-friedrich/OASIS/blob/e62063cfd8bc0f06625aebd3ea3e09133665b409/oasis/functions.py#L17
  '''
  spikes = spike_trains.astype(np.float32)

  for i in range(2, hparams.duration):
    if len(g) == 2:
      spikes[:, i] += g[0] * spikes[:, i - 1] + g[1] * spikes[:, i - 2]
    else:
      spikes[:, i] += g[0] * spikes[:, i - 1]

  signals = b + spikes + sn * np.random.randn(hparams.num_neurons,
                                              hparams.duration)

  return signals.astype(np.float32)


def main(hparams):
  mean, covariance = get_recorded_data_statistics(hparams)

  dg_spikes = generate_dg_spikes(hparams, mean, covariance)

  dg_signals = spikes_to_signals(hparams, dg_spikes)

  if os.path.exists(hparams.output):
    os.remove(hparams.output)

  with open(hparams.output, 'wb') as file:
    pickle.dump({'signals': dg_signals, 'oasis': dg_spikes}, file)

  print('Saved {} DG signals and spikes to {}'.format(
      len(dg_signals), hparams.output))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input', default='raw_data/ST260_Day4_signals4Bryan.pkl', type=str)
  parser.add_argument('--output', default='dg/data.pkl', type=str)
  hparams = parser.parse_args()
  main(hparams)
