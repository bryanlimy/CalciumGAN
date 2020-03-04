import numpy as np
import tensorflow as tf
import quantities as pq
from neo.core import SpikeTrain
from oasis.oasis_methods import oasisAR1


def train_to_neo(train, t_stop=None):
  ''' convert a single spike train to Neo SpikeTrain in sec scale '''
  return SpikeTrain(
      np.nonzero(train)[0] * pq.ms,
      units=pq.s,
      t_stop=train.shape[-1] * pq.ms if t_stop is None else t_stop,
      dtype=np.float32)


def trains_to_neo(trains):
  ''' convert array of spike trains to list of  Neo SpikeTrains in sec scale '''
  assert trains.ndim == 2
  t_stop = trains.shape[-1] * pq.ms
  return [train_to_neo(trains[i], t_stop=t_stop) for i in range(len(trains))]


def oasis_function(signal, threshold=0.5):
  ''' apply OASIS function to a single calcium signal and binarize spike train 
  with threshold '''
  if signal.dtype != np.double:
    signal = signal.astype(np.double)
  _, train = oasisAR1(signal, g=0.95, s_min=.55)
  return np.where(train > threshold, 1.0, 0.0)


def deconvolve_signals(signals, threshold=0.5, to_neo=False):
  ''' apply OASIS function to array of signals and convert to Neo SpikeTrain 
  if to_neo is True'''
  if tf.is_tensor(signals):
    signals = signals.numpy()

  assert signals.ndim == 2

  if signals.dtype != np.double:
    signals = signals.astype(np.double)

  spike_trains = []
  t_stop = signals.shape[-1] * pq.ms

  for i in range(len(signals)):
    spike_train = oasis_function(signals[i], threshold=threshold)
    spike_trains.append(
        train_to_neo(spike_train, t_stop=t_stop) if to_neo else spike_train)

  if not to_neo:
    spike_trains = np.array(spike_trains, dtype=np.float32)

  return spike_trains
