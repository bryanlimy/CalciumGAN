import numpy as np
import tensorflow as tf
from multiprocessing import Pool

from . import utils
from oasis.oasis_methods import oasisAR1
from neo.core import SpikeTrain
import quantities as pq
from . import h5_helpers
from . import spike_helper
from time import time
import pickle


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


def deconvolve_saved_signals(hparams, filename):
  start = time()
  with h5_helpers.open_h5(filename, mode='a') as file:
    fake_signals = file['fake_signals'][:]
    fake_spikes = deconvolve_signals(
        fake_signals, num_processors=hparams.num_processors)

    file.create_dataset(
        'fake_spikes',
        dtype=fake_spikes.dtype,
        data=fake_spikes,
        chunks=True,
        maxshape=(None, fake_spikes.shape[1], fake_spikes.shape[2]))
  elapse = time() - start

  if hparams.verbose:
    print('Deconvolve {} signals in {:.2f}s'.format(len(fake_spikes), elapse))


def numpy_to_neo_trains(spikes):
  shape = spikes.shape

  if len(shape) > 2:
    spikes = np.reshape(spikes, newshape=(shape[0] * shape[1], shape[2]))

  neo_trains = [
      SpikeTrain(np.nonzero(spikes[i])[0], units=pq.ms, t_stop=shape[-1])
      for i in range(len(spikes))
  ]

  return neo_trains


def compute_spike_metrics(real_signals, real_spikes, fake_signals, fake_spikes):
  assert type(real_spikes) is None or type(real_spikes) == list
  assert type(fake_spikes) is None or type(fake_spikes) == list

  if real_spikes is None:
    real_spikes = deconvolve_signals(real_signals, numpy_to_neo_trains=8)
    real_spikes = numpy_to_neo_trains(real_spikes)

  if fake_spikes is None:
    fake_spikes = deconvolve_signals(fake_signals)
