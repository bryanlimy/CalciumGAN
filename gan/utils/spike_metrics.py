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


def correlation_coefficients(spikes1, spikes2, binsize=500 * pq.ms):
  result = elephant.spike_train_correlation.corrcoef(
      elephant.conversion.BinnedSpikeTrain(spikes1 + spikes2, binsize=binsize))
  return np.mean(result[len(spikes1):, :len(spikes2)])


def covariance(spikes1, spikes2, binsize=500 * pq.ms):
  result = elephant.spike_train_correlation.covariance(
      elephant.conversion.BinnedSpikeTrain(spikes1 + spikes2, binsize=binsize))
  return np.mean(result[len(spikes1):, :len(spikes2)])


def van_rossum_distance(spikes1, spikes2):
  ''' return the mean van rossum distance between spikes1 and spikes2 '''
  result = elephant.spike_train_dissimilarity.van_rossum_dist(spikes1 + spikes2)
  return np.mean(result[len(spikes1):, :len(spikes2)], dtype=np.float32)


def victor_purpura_distance(spikes1, spikes2):
  result = elephant.spike_train_dissimilarity.victor_purpura_dist(spikes1 +
                                                                  spikes2)
  return np.mean(result[len(spikes1):, :len(spikes2)])
