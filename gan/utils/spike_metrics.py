import elephant
import numpy as np
import quantities as pq


def mean_firing_rate(spikes):
  ''' get mean firing rate of spikes in Hz'''
  result = [
      elephant.statistics.mean_firing_rate(spikes[i])
      for i in range(len(spikes))
  ]
  return np.array(result, dtype=np.float32)


def correlation_coefficients(spikes1, spikes2, binsize=100 * pq.ms):
  spikes = spikes1 + spikes2 if spikes2 is not None else spikes1
  binned = elephant.conversion.BinnedSpikeTrain(spikes, binsize=binsize)

  result = elephant.spike_train_correlation.corrcoef(binned)

  if spikes2 is not None:
    result = result[len(spikes1):, :len(spikes2)]

  return result


def covariance(spikes1, spikes2, binsize=100 * pq.ms):
  spikes = spikes1 + spikes2 if spikes2 is not None else spikes1
  binned = elephant.conversion.BinnedSpikeTrain(spikes, binsize=binsize)

  result = elephant.spike_train_correlation.covariance(binned)

  if spikes2 is not None:
    result = result[len(spikes1):, :len(spikes2)]

  return result


def van_rossum_distance(spikes1, spikes2):
  ''' return the mean van rossum distance between spikes1 and spikes2 '''
  spikes = spikes1 + spikes2 if spikes2 is not None else spikes1

  result = elephant.spike_train_dissimilarity.van_rossum_dist(spikes)

  if spikes2 is not None:
    result = result[len(spikes1):, :len(spikes2)]

  return result


def victor_purpura_distance(spikes1, spikes2):
  spikes = spikes1 + spikes2 if spikes2 is not None else spikes1

  result = elephant.spike_train_dissimilarity.victor_purpura_dist(spikes)

  if spikes2 is not None:
    result = result[len(spikes1):, :len(spikes2)]

  return result
