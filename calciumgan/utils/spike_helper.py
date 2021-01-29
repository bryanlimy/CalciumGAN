import os
import elephant
import numpy as np
import quantities as pq
from neo.core import SpikeTrain

from calciumgan.utils import utils
from calciumgan.utils.cascade.spike_inference import spike_inference


def deconvolve_samples(hparams):
  if not hasattr(hparams, 'spikes_filename'):
    hparams.spikes_filename = os.path.join(hparams.samples_dir, 'spikes.h5')

  spike_inference(
      signals_filename=hparams.signals_filename,
      spikes_filename=hparams.spikes_filename)

  utils.update_json(
      filename=os.path.join(hparams.output_dir, 'hparams.json'),
      data={'spikes_filename': hparams.spikes_filename})

  if hparams.verbose:
    print(f'saved inferred spike trains to {hparams.spikes_filename}')


def get_spike_times(spike_rates, threshold=0.25):
  return np.where(spike_rates > threshold)


def get_spike_trains(spike_rates):
  spike_trains = np.zeros(spike_rates.shape, dtype=np.int8)
  spike_trains[get_spike_times(spike_rates)] = 1.0
  return spike_trains


def train_to_neo(spike_rate, frame_rate=24.0):
  ''' convert a single spike train to Neo SpikeTrain '''
  spike_time = get_spike_times(spike_rate)[0]
  spike_time = (spike_time / frame_rate) * pq.s
  t_stop = (len(spike_rate) / frame_rate) * pq.s
  return SpikeTrain(spike_time, units=pq.s, t_stop=t_stop, dtype=np.float32)


def trains_to_neo(trains):
  ''' convert array of spike trains to list of  Neo SpikeTrains in sec scale '''
  assert trains.ndim == 2
  return [train_to_neo(trains[i]) for i in range(len(trains))]


def mean_firing_rate(spikes):
  ''' get mean firing rate of spikes in Hz'''
  result = [
      elephant.statistics.mean_firing_rate(spikes[i]) * pq.s
      for i in range(len(spikes))
  ]
  return np.array(result, dtype=np.float32)


def correlation_coefficients(spikes1, spikes2, binsize=500 * pq.ms):
  spikes = spikes1 + spikes2 if spikes2 is not None else spikes1
  binned = elephant.conversion.BinnedSpikeTrain(spikes, binsize=binsize)
  result = elephant.spike_train_correlation.correlation_coefficient(
      binned, binary=True, fast=False)
  if spikes2 is not None:
    result = result[len(spikes1):, :len(spikes2)]
  return result.astype(np.float32)


def covariance(spikes1, spikes2, binsize=500 * pq.ms):
  spikes = spikes1 + spikes2 if spikes2 is not None else spikes1
  binned = elephant.conversion.BinnedSpikeTrain(spikes, binsize=binsize)
  result = elephant.spike_train_correlation.covariance(
      binned, binary=True, fast=False)
  if spikes2 is not None:
    result = result[len(spikes1):, :len(spikes2)]
  return result.astype(np.float32)


def van_rossum_distance(spikes1, spikes2):
  ''' return the mean van rossum distance between spikes1 and spikes2 '''
  spikes = spikes1 + spikes2 if spikes2 is not None else spikes1
  result = elephant.spike_train_dissimilarity.van_rossum_distance(spikes)
  if spikes2 is not None:
    result = result[len(spikes1):, :len(spikes2)]
  return result.astype(np.float32)


def victor_purpura_distance(spikes1, spikes2):
  spikes = spikes1 + spikes2 if spikes2 is not None else spikes1
  result = elephant.spike_train_dissimilarity.victor_purpura_distance(spikes)
  if spikes2 is not None:
    result = result[len(spikes1):, :len(spikes2)]
  return result.astype(np.float32)
