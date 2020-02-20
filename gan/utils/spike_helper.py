import numpy as np
from time import time
from tqdm import tqdm
import tensorflow as tf
import quantities as pq
from neo.core import SpikeTrain
from oasis.oasis_methods import oasisAR1
from multiprocessing import Manager, Pool

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

  return np.reshape(spikes, newshape=shape)


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
  ''' convert numpy array to Neo SpikeTrain in sec scale '''
  if type(array) == list and type(array[0]) == SpikeTrain:
    return array
  shape = array.shape
  if len(shape) > 2:
    array = np.reshape(array, newshape=(shape[0] * shape[1], shape[2]))

  return [
      SpikeTrain(
          np.nonzero(array[i])[0] * pq.ms,
          units=pq.s,
          t_stop=shape[-1] * pq.ms,
          dtype=np.float32) for i in range(len(array))
  ]


def count_occurrence(array):
  unique, count = np.unique(array, return_counts=True)
  return dict(zip(unique, count))


def firing_rate_histogram(real_firing_rate, fake_firing_rate):
  occurrence1 = count_occurrence(real_firing_rate)
  occurrence2 = count_occurrence(fake_firing_rate)


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

  real_firing_rate = spike_metrics.mean_firing_rate(real_spikes)
  fake_firing_rate = spike_metrics.mean_firing_rate(fake_spikes)
  firing_rate_error = np.mean(np.square(real_firing_rate - fake_firing_rate))
  utils.add_to_dict(metrics, 'spike_metrics/firing_rate_error',
                    firing_rate_error)

  # corrcoef = spike_metrics.correlation_coefficients(real_spikes, fake_spikes)
  # utils.add_to_dict(metrics, 'spike_metrics/cross_coefficient', corrcoef)

  # covariance = spike_metrics.covariance(real_spikes, fake_spikes)
  # utils.add_to_dict(metrics, 'spike_metrics/covariance', covariance)

  van_rossum_distance = spike_metrics.van_rossum_distance(
      real_spikes, fake_spikes)
  utils.add_to_dict(metrics, 'spike_metrics/van_rossum_distance',
                    van_rossum_distance)


def measure_spike_metrics_from_file(metrics, filename, neuron):
  """ measure spike metrics for neuron in file and write results to metrics """
  with h5_helper.open_h5(filename, mode='r') as file:
    real_signals = file['real_signals'][neuron]
    fake_signals = file['fake_signals'][neuron]
    real_spikes = file['real_spikes'][neuron]

  measure_spike_metrics(
      metrics,
      real_signals=real_signals,
      fake_signals=fake_signals,
      real_spikes=real_spikes,
      fake_spikes=None)


def rearrange_saved_signals(hparams, filename):
  ''' rearrange signals to (neurons, samples, segments) '''
  if hparams.verbose:
    print('\tRearrange saved signals')
  shape = (hparams.validation_size, hparams.num_neurons)
  with h5_helper.open_h5(filename, mode='r+') as file:
    for key in file.keys():
      value = file[key][:]
      # check if value has shape (samples, neurons)
      if value.shape[:2] == shape:
        value = np.swapaxes(value, axis1=0, axis2=1)
        h5_helper.overwrite_dataset(file, key, value)


def record_spike_metrics(hparams, epoch, summary):
  if hparams.verbose:
    print('Measuring spike metrics...')

  start = time()

  filename = utils.get_signal_filename(hparams, epoch)

  rearrange_saved_signals(hparams, filename)

  if hparams.num_processors > 1:
    manager = Manager()
    metrics = manager.dict()

    pool = Pool(processes=hparams.num_processors)
    pool.starmap(
        measure_spike_metrics_from_file,
        [(metrics, filename, neuron) for neuron in range(hparams.num_neurons)])
    pool.close()
  else:
    metrics = {}
    for neuron in tqdm(
        range(hparams.num_neurons),
        desc='\tNeuron',
        disable=not bool(hparams.verbose)):
      measure_spike_metrics_from_file(metrics, filename, neuron)

  end = time()

  summary.scalar('elapse/spike_metrics', end - start, training=False)

  for tag, value in metrics.items():
    if tag.startswith('spike_metrics'):
      if hparams.verbose:
        print('\t{}: {:.04f}'.format(tag, np.mean(value)))
      summary.scalar(tag, np.mean(value), training=False)
