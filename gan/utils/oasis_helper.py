import numpy as np
import tensorflow as tf
from multiprocessing import Pool

from oasis.oasis_methods import oasisAR1


def split(sequence, n):
  """ divide sequence into n sub-sequence evenly"""
  k, m = divmod(len(sequence), n)
  return [
      sequence[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)
  ]


def _deconvolve_signals(signals):
  spikes = np.zeros(signals.shape)
  for i in range(len(signals)):
    _, spikes[i] = oasisAR1(signals[i], g=0.95, s_min=.55)
  spikes = np.where(spikes > 0.5, 1.0, 0.0)
  return spikes


def deconvolve_signals(signals, to_tensor=False, num_processors=1):
  if tf.is_tensor(signals):
    signals = signals.numpy()

  shape = signals.shape
  if len(shape) > 2:
    signals = np.reshape(signals, newshape=(shape[0] * shape[1], shape[2]))
  signals = signals.astype('double')

  if num_processors > 2:
    num_jobs = min(len(signals), num_processors)
    subsets = split(signals, n=num_jobs)
    pool = Pool(processes=num_jobs)
    spikes = pool.map(_deconvolve_signals, subsets)
    pool.close()
    spikes = np.concatenate(spikes, axis=0)
  else:
    spikes = np.array(_deconvolve_signals(signals))

  assert spikes.shape == signals.shape

  spikes = np.reshape(spikes, newshape=shape)

  return tf.convert_to_tensor(spikes, dtype=tf.float32) if to_tensor else spikes
