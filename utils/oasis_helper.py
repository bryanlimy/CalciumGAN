import numpy as np
import tensorflow as tf
from oasis.functions import deconvolve
from multiprocessing import Pool, cpu_count


def split(sequence, n):
  """ divide sequence into n sub-sequence evenly"""
  k, m = divmod(len(sequence), n)
  return [
      sequence[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)
  ]


def _deconvolve_signals(signals):
  spikes = []
  for i in range(len(signals)):
    c, s, b, g, lam = deconvolve(signals[i], g=(None,), penalty=1)
    spikes.append(s / s.max() if s.max() > 0 else s)
  return spikes


def deconvolve_signals(signals, to_tensor=False, multiprocessing=True):
  if tf.is_tensor(signals):
    signals = signals.numpy()

  signals = signals.astype('double')

  if multiprocessing:
    num_jobs = min(len(signals), cpu_count() - 6)
    subsets = split(signals, n=num_jobs)
    pool = Pool(processes=num_jobs)
    spikes = pool.map(_deconvolve_signals, subsets)
    pool.close()
    spikes = np.concatenate(spikes, axis=0)
  else:
    spikes = np.array(_deconvolve_signals(signals))

  assert spikes.shape == signals.shape

  return tf.convert_to_tensor(spikes, dtype=tf.float32) if to_tensor else spikes
