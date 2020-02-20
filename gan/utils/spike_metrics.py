import elephant
import numpy as np
import quantities as pq


def mean_firing_rate(spikes):
  ''' get mean firing rate of spikes in Hz'''
  rates = [
      elephant.statistics.mean_firing_rate(spikes[i])
      for i in range(len(spikes))
  ]
  return np.array(rates, dtype=np.float32)


def van_rossum_distance(spikes1, spikes2):
  assert len(spikes1) == len(spikes2)
  distance_matrix = elephant.spike_train_dissimilarity.van_rossum_dist(spikes1 +
                                                                       spikes2)
  return np.diag(distance_matrix[len(spikes1):])


def victor_purpura_distance(spikes1, spikes2):
  assert len(spikes1) == len(spikes2)
  distance_matrix = elephant.spike_train_dissimilarity.victor_purpura_dist(
      spikes1 + spikes2)
  return np.diag(distance_matrix[len(spikes1):])


def correlation_coefficients(spikes1, spikes2, binsize=5 * pq.ms):
  assert len(spikes1) == len(spikes2)
  corrcoef_matrix = elephant.spike_train_correlation.corrcoef(
      elephant.conversion.BinnedSpikeTrain(spikes1 + spikes2, binsize=binsize),
      fast=False)
  return np.diag(corrcoef_matrix[len(spikes1):])


def covariance(spikes1, spikes2, binsize=5 * pq.ms):
  assert len(spikes1) == len(spikes2)
  covariance_matrix = elephant.spike_train_correlation.covariance(
      elephant.conversion.BinnedSpikeTrain(spikes1 + spikes2, binsize=binsize),
      fast=False)
  return np.diag(covariance_matrix[len(spikes1):])
