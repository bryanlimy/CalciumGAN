import numpy as np
import tensorflow as tf


def mean_spike_count(spikes):
  binarized = (spikes > np.random.random(spikes.shape)).astype(np.float32)
  firing_rate_per_trace = np.mean(binarized, axis=-1)
  mean_spike_per_batch = np.mean(firing_rate_per_trace, axis=-1)
  return mean_spike_per_batch


def derivative_mse(set1, set2):
  diff1 = np.diff(set1, n=1, axis=-1)
  diff2 = np.diff(set2, n=1, axis=-1)
  mse = np.mean(np.square(diff1 - diff2))
  return mse


class ExponentialDecay():
  """
  Exponentially decaying function with additive method.
  Useful for efficiently computing Van Rossum distance.
  """

  def __init__(self, k=None, dt=0.0001):
    self.value = 0.0
    self.dt = dt
    self.k = k
    self.decay_factor = np.exp(-dt * k)

  def update(self):
    self.value = self.value * self.decay_factor
    return self.value

  def spike(self):
    self.value += 1
    return self.value

  def reset(self):
    self.value = 0


def van_rossum_distance(spike1, spike2, tc=1000, bin_width=0.0001, t_extra=1):
  """
  Calculates the Van Rossum distance between spike trains
  Note that the default parameters are optimized for inputs in units of seconds.
  :param spike1: array of spike times for first spike train
  :param spike2: array of spike times for second spike train
  :param bin_width: precision in units of time to compute integral
  :param t_extra: how much beyond max time do we keep integrating until?
  """

  # by default, we assume spike times are in seconds,
  # keep integrating up to 0.5 s past last spike
  t_max = max([spike1[-1], spike2[-1]]) + t_extra

  # t_min = min(st_0[0],st_0[0])
  t_range = np.arange(0, t_max, bin_width)

  # we use a spike induced current to perform the computation
  decay = ExponentialDecay(k=1.0 / tc, dt=bin_width)

  f_0 = t_range * 0.0
  f_1 = t_range * 0.0

  # we make copies of these arrays, since we are going to "pop" them
  s_0 = list(spike1[:])
  s_1 = list(spike2[:])

  for (st, f) in [(s_0, f_0), (s_1, f_1)]:
    # set the internal value to zero
    decay.reset()

    for (t_ind, t) in enumerate(t_range):
      f[t_ind] = decay.update()
      if len(st) > 0:
        if t > st[0]:
          f[t_ind] = decay.spike()
          st.pop(0)

  distance = np.sqrt((bin_width / tc) * np.linalg.norm((f_0 - f_1), 1))
  return distance
