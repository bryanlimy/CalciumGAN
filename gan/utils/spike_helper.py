import numpy as np
import tensorflow as tf
from multiprocessing import Pool

from . import utils
from oasis.oasis_methods import oasisAR1
from neo.core import SpikeTrain
import quantities as pq


def _deconvolve_signals(signals):
  spikes = np.zeros(signals.shape)
  for i in range(len(signals)):
    _, spikes[i] = oasisAR1(signals[i], g=0.95, s_min=.55)
  spikes = np.where(spikes > 0.5, 1.0, 0.0)
  return spikes


def deconvolve_signals(signals, num_processors=1):
  if tf.is_tensor(signals):
    signals = signals.numpy()

  shape = signals.shape
  if len(shape) > 2:
    signals = np.reshape(signals, newshape=(shape[0] * shape[1], shape[2]))

  signals = signals.astype('double')

  if num_processors > 2:
    num_jobs = min(len(signals), num_processors)
    subsets = utils.split(signals, n=num_jobs)
    pool = Pool(processes=num_jobs)
    spikes = pool.map(_deconvolve_signals, subsets)
    pool.close()
    spikes = np.concatenate(spikes, axis=0)
  else:
    spikes = np.array(_deconvolve_signals(signals))

  spikes = np.reshape(spikes, newshape=shape)

  return spikes


def numpy_to_neo(spikes):
  shape = spikes.shape

  if len(shape) > 2:
    spikes = np.reshape(spikes, newshape=(shape[0] * shape[1], shape[2]))

  spike_times = np.nonzero(spikes)

  neo_trains = np.array([
      SpikeTrain(spike_times[i], units=pq.ms, t_stop=shape[-1])
      for i in range(len(spike_times))
  ])

  if len(shape) > 2:
    neo_trains = np.reshape(neo_trains, newshape=(shape[0], shape[1]))

  return neo_trains
