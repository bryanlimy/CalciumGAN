import numpy as np
import tensorflow as tf
from multiprocessing import Pool
import elephant

from . import utils


def mean_spike_count(spikes):
  spike_count = np.sum(spikes, axis=-1)
  mean_spike_count = np.mean(spike_count, axis=0)
  return mean_spike_count


def mean_spike_count_error(spike1, spike2):
  count1 = mean_spike_count(spike1)
  count2 = mean_spike_count(spike2)
  return np.mean(np.square(count1 - count2))


def mean_firing_rate(spikes):
  firing_rate = np.sum(spikes, axis=-1) / spikes.shape[-1]
  mean_firing_rate = np.mean(firing_rate, axis=0)
  return mean_firing_rate


def mean_firing_rate_error(spike1, spike2):
  rate1 = mean_firing_rate(spike1)
  rate2 = mean_firing_rate(spike2)
  return np.mean(np.square(rate1 - rate2))


def derivative_mse(set1, set2):
  diff1 = np.diff(set1, n=1, axis=-1)
  diff2 = np.diff(set2, n=1, axis=-1)
  mse = np.mean(np.square(diff1 - diff2))
  return mse


def van_rossum_distance(real_spikes, fake_spikes):
  assert len(real_spikes) == len(fake_spikes)
  distance_matrix = elephant.spike_train_dissimilarity.van_rossum_dist(
      real_spikes + fake_spikes)
  distance = np.diag(distance_matrix[len(real_spikes):])
  return np.mean(distance)


def victor_purpura_distance(real_spikes, fake_spikes):
  assert len(real_spikes) == len(fake_spikes)
  distance_matrix = elephant.spike_train_dissimilarity.victor_purpura_dist(
      real_spikes + fake_spikes)
  distance = np.diag(distance_matrix[len(real_spikes):])
  return np.mean(distance)
