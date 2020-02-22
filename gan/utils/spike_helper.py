import numpy as np
from time import time
import tensorflow as tf
import quantities as pq
from neo.core import SpikeTrain
from oasis.oasis_methods import oasisAR1
from multiprocessing import Manager, Pool

from . import utils
from . import h5_helper
from . import spike_metrics


def oasis_function(signals, threshold=0.5):
  if signals.dtype != np.double:
    signals = signals.astype(np.double)
  spikes = np.zeros(signals.shape, dtype=np.float32)
  for i in range(len(signals)):
    _, spikes[i] = oasisAR1(signals[i], g=0.95, s_min=.55)
  return np.where(spikes > threshold, 1.0, 0.0)


def deconvolve_signals(signals, num_processors=1):
  if tf.is_tensor(signals):
    signals = signals.numpy()

  shape = signals.shape
  if len(shape) > 2:
    signals = np.reshape(signals, newshape=(shape[0] * shape[1], shape[2]))

  if num_processors > 2:
    num_jobs = min(len(signals), num_processors)
    pool = Pool(processes=num_jobs)
    spikes = pool.map(oasis_function, utils.split(signals, n=num_jobs))
    pool.close()
    spikes = np.concatenate(spikes, axis=0)
  else:
    spikes = np.array(oasis_function(signals), dtype=np.float32)

  return np.reshape(spikes, newshape=shape)


def rearrange_saved_signals(hparams, filename):
  ''' rearrange signals to (neurons, samples, segments) '''
  shape = (hparams.validation_size, hparams.num_neurons)
  with h5_helper.open_h5(filename, mode='r+') as file:
    for key in file.keys():
      value = file[key][:]
      # check if value has shape (samples, neurons)
      if value.shape[:2] == shape:
        value = np.swapaxes(value, axis1=0, axis2=1)
        h5_helper.overwrite_dataset(file, key, value)


def numpy_to_neo_trains(array):
  ''' convert numpy array to Neo SpikeTrain in sec scale '''
  if type(array) == list and type(array[0]) == SpikeTrain:
    return array
  assert array.ndim == 2
  t_stop = array.shape[-1] * pq.ms
  return [
      SpikeTrain(
          np.nonzero(array[i])[0] * pq.ms,
          units=pq.s,
          t_stop=t_stop,
          dtype=np.float32) for i in range(len(array))
  ]


def count_occurrence(array):
  unique, count = np.unique(array, return_counts=True)
  return dict(zip(unique, count))


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def compute_spike_metrics(metrics,
                          neuron,
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
  metrics['spike_metrics/firing_rate_error'][neuron] = firing_rate_error

  metrics['firing_rate'] = (real_firing_rate, fake_firing_rate)

  corrcoef = spike_metrics.correlation_coefficients(real_spikes, fake_spikes)
  metrics['spike_metrics/cross_coefficient'][neuron] = corrcoef

  covariance = spike_metrics.covariance(real_spikes, fake_spikes)
  metrics['spike_metrics/covariance'][neuron] = covariance

  # compares to first 1000 samples to save time
  van_rossum_distance = spike_metrics.van_rossum_distance(
      real_spikes[:1000], fake_spikes[:1000])
  metrics['spike_metrics/van_rossum_distance'][neuron] = van_rossum_distance


def neuron_spike_metrics(metrics, filename, neuron):
  """ measure spike metrics for neuron in file and write results to metrics """
  with h5_helper.open_h5(filename, mode='r') as file:
    real_signals = file['real_signals'][neuron]
    fake_signals = file['fake_signals'][neuron]
    real_spikes = file['real_spikes'][neuron]

  compute_spike_metrics(
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
  rearrange_saved_signals(hparams, filename)

  manager = Manager()
  metrics = manager.dict() if hparams.num_processors > 1 else dict()

  metrics['spike_metrics/firing_rate_error'] = [None] * hparams.num_neurons
  metrics['spike_metrics/cross_coefficient'] = [None] * hparams.num_neurons
  metrics['spike_metrics/covariance'] = [None] * hparams.num_neurons
  metrics['spike_metrics/van_rossum_distance'] = [None] * hparams.num_neurons
  metrics['firing_rate'] = [None] * hparams.num_neurons

  if hparams.num_processors > 1:
    pool = Pool(processes=hparams.num_processors)
    pool.starmap(
        neuron_spike_metrics,
        [(metrics, filename, neuron) for neuron in range(hparams.num_neurons)])
    pool.close()
  else:
    for neuron in range(hparams.num_neurons):
      neuron_spike_metrics(metrics, filename, neuron)

  end = time()

  summary.scalar('elapse/spike_metrics', end - start, training=False)

  for key, value in metrics.items():
    result = np.mean(value)
    if key.startswith('spike_metrics'):
      if hparams.verbose:
        print('\t{}: {:.04f}'.format(key, np.mean(result)))
      summary.scalar(key, result, training=False)
    elif key == 'firing_rates':
      for i, firing_rates in enumerate(value):
        summary.plot_histogram(
            'firing_rate/neuron_{}'.format(i),
            firing_rates,
            xlabel='Hz',
            ylabel='Count',
            training=False)
