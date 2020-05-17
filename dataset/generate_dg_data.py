import os
import pickle
import elephant
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import quantities as pq
from neo.core import SpikeTrain

from dg.dichot_gauss import DichotGauss


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
  neo_trains = trains_to_neo(spike_trains)

  hparams.num_neurons = len(neo_trains)

  mean = mean_firing_rate(neo_trains)
  corr = correlation_coefficients(neo_trains)
  return mean, corr


def generate_dg_spikes(hparams, mean, corr):
  dg = DichotGauss(
      hparams.num_neurons,
      mean=np.expand_dims(mean, axis=0),
      corr=corr,
      make_pd=True)
  spike_trains = dg.sample(repeats=hparams.duration)

  # reshape to (num_neurons, duration)
  spike_trains = np.reshape(
      spike_trains, newshape=(hparams.num_neurons, hparams.duration))

  return spike_trains.astype(np.float32)


def spikes_to_signals(hparams, spike_trains, g=[.95], sn=.3, b=0):
  ''' Convert spike trains to calcium signals 
  Code extracted from https://github.com/j-friedrich/OASIS/blob/e62063cfd8bc0f06625aebd3ea3e09133665b409/oasis/functions.py#L17
  '''
  truth = spike_trains.astype(np.float32)

  for i in range(2, hparams.duration):
    if len(g) == 2:
      truth[:, i] += g[0] * truth[:, i - 1] + g[1] * truth[:, i - 2]
    else:
      truth[:, i] += g[0] * truth[:, i - 1]

  signals = b + truth + sn * np.random.randn(hparams.num_neurons,
                                             hparams.duration)

  return signals.astype(np.float32)


def main(hparams):
  mean, corr = get_recorded_data_statistics(hparams)

  dg_spikes = generate_dg_spikes(hparams, mean, corr)

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
  parser.add_argument('--duration', default=21556, type=int)
  hparams = parser.parse_args()
  main(hparams)
