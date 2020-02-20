import numpy as np
from time import time
import tensorflow as tf
import quantities as pq
from neo.core import SpikeTrain
from oasis.oasis_methods import oasisAR1
from multiprocessing import Process, Manager, Pool

from . import utils
from . import h5_helper
from . import spike_metrics


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
  with h5_helper.open_h5(filename, mode='a') as file:
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


def numpy_to_neo_trains(array):
  if type(array) == list and type(array[0]) == SpikeTrain:
    return array
  shape = array.shape
  if len(shape) > 2:
    array = np.reshape(array, newshape=(shape[0] * shape[1], shape[2]))

  return [
      SpikeTrain(np.nonzero(array[i])[0], units=pq.ms, t_stop=shape[-1])
      for i in range(len(array))
  ]


def measure_spike_metrics(metrics,
                          real_signals,
                          fake_signals,
                          real_spikes=None,
                          fake_spikes=None):
  if real_spikes is None:
    real_spikes = deconvolve_signals(real_signals)
  real_spikes = numpy_to_neo_trains(real_spikes)

  if fake_spikes is None:
    fake_spikes = deconvolve_signals(fake_signals)
  fake_spikes = numpy_to_neo_trains(fake_spikes)

  firing_rate_error = spike_metrics.mean_firing_rate_error(
      real_spikes, fake_spikes)
  utils.add_to_dict(metrics, 'spike_metrics/firing_rate_error',
                    firing_rate_error)

  corrcoef = spike_metrics.correlation_coefficients(real_spikes, fake_spikes)
  utils.add_to_dict(metrics, 'spike_metrics/cross_coefficient', corrcoef)

  covariance = spike_metrics.covariance(real_spikes, fake_spikes)
  utils.add_to_dict(metrics, 'spike_metrics/covariance', covariance)


def measure_spike_metrics_from_file(metrics, filename, index=(0, None)):
  """ measure spike metrics of content within (start, end) range in filename 
  and write results to metrics """
  with h5_helper.open_h5(filename, mode='r') as file:
    real_signals = file['real_signals'][index[0]:index[1]]
    fake_signals = file['fake_signals'][index[0]:index[1]]
    real_spikes = file['real_spikes'][index[0]:index[1]]

  measure_spike_metrics(
      metrics,
      real_signals=real_signals,
      fake_signals=fake_signals,
      real_spikes=real_spikes,
      fake_spikes=None)


def record_spike_metrics(hparams, epoch, summary):
  if hparams.verbose:
    print('Measuring spike metrics...')

  start = time()

  filename = utils.get_signal_filename(hparams, epoch)

  if hparams.num_processors > 1:
    length = h5_helper.dataset_length(filename, 'real_signals')

    num_jobs = min(length, hparams.num_processors)
    indexes = utils.split_index(length, n=num_jobs)

    manager = Manager()
    metrics = manager.dict()

    jobs = []
    for i in range(num_jobs):
      job = Process(
          target=measure_spike_metrics_from_file,
          args=(metrics, filename, indexes[i]))
      jobs.append(job)
      job.start()
    for job in jobs:
      job.join()
  else:
    metrics = {}
    measure_spike_metrics_from_file(metrics, filename)

  end = time()

  summary.scalar('elapse/spike_metrics', end - start, training=False)

  for tag, value in metrics.items():
    if hparams.verbose:
      print('\t{}: {:.04f}'.format(tag, np.mean(value)))
    summary.scalar(tag, np.mean(value), training=False)
