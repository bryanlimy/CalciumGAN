import elephant
import numpy as np
import quantities as pq


def mean_firing_rate(spikes):
  return np.array([
      elephant.statistics.mean_firing_rate(spikes[i])
      for i in range(len(spikes))
  ])


def mean_firing_rate_error(spikes1, spikes2):
  spikes1_firing_rate = mean_firing_rate(spikes1)
  spikes2_firing_rate = mean_firing_rate(spikes2)
  return np.abs(np.mean(spikes1_firing_rate) - np.mean(spikes2_firing_rate))


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
