import pickle
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


def deconvolve_and_neo(signals, threshold=0.5):
  assert signals.ndim == 2
  if signals.dtype != np.double:
    signals = signals.astype(np.double)
  spikes = []
  for i in range(len(signals)):
    _, spike = oasisAR1(signals[i], g=0.95, s_min=.55)
    spike = np.where(spike > threshold, 1.0, 0.0)
    spike = SpikeTrain(
        np.nonzero(spike)[0] * pq.ms,
        units=pq.s,
        t_stop=spike.shape[-1] * pq.ms,
        dtype=np.float32)
    spikes.append(spike)
  return spikes


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


def neuron_spike_metrics(hparams, epoch, neuron, metrics):
  """ measure spike metrics for neuron in file and write results to metrics """
  # get real neuron data
  with open(utils.get_fake_filename(hparams, neuron), 'rb') as file:
    data = pickle.load(file)
  real_spikes = data['real_spikes']
  assert type(real_spikes) == list and type(real_spikes[0]) == SpikeTrain

  # get fake neuron data
  fake_filename = utils.get_fake_filename(hparams, epoch)
  with h5_helper.open_h5(fake_filename, mode='r') as file:
    fake_signals = file['fake_signals'][neuron]
  fake_spikes = deconvolve_and_neo(fake_signals)

  assert len(real_spikes) == len(fake_spikes)

  if 'spike_metrics/firing_rate_error' in metrics:
    real_firing_rate = spike_metrics.mean_firing_rate(real_spikes)
    fake_firing_rate = spike_metrics.mean_firing_rate(fake_spikes)
    firing_rate_error = np.mean(np.square(real_firing_rate - fake_firing_rate))
    metrics['spike_metrics/firing_rate_error'][neuron] = firing_rate_error

    if 'histogram/firing_rate' in metrics:
      metrics['histogram/firing_rate'][neuron] = (real_firing_rate,
                                                  fake_firing_rate)
  if 'spike_metrics/cross_coefficient' in metrics:
    corrcoef = spike_metrics.correlation_coefficients(real_spikes, fake_spikes)
    metrics['spike_metrics/cross_coefficient'][neuron] = corrcoef

  if 'spike_metrics/covariance' in metrics:
    covariance = spike_metrics.covariance(real_spikes, fake_spikes)
    metrics['spike_metrics/covariance'][neuron] = covariance

  if 'spike_metrics/van_rossum_distance' in metrics:
    # compares to first 1000 samples to save time
    van_rossum_distance = spike_metrics.van_rossum_distance(
        real_spikes[:1000], fake_spikes[:1000])
    metrics['spike_metrics/van_rossum_distance'][neuron] = van_rossum_distance


def populate_metrics_dict(num_processors, num_neurons):
  ''' create thread-safe dictionary to store metrics '''
  keys = [
      'spike_metrics/firing_rate_error', 'histogram/firing_rate',
      'spike_metrics/cross_coefficient', 'spike_metrics/covariance',
      'spike_metrics/van_rossum_distance'
  ]
  if num_processors == 1:
    metrics = {key: [None] * num_neurons for key in keys}
  else:
    manager = Manager()
    metrics = manager.dict(
        {key: manager.list([None] * num_neurons) for key in keys})
  return metrics


def record_spike_metrics(hparams, epoch, summary):
  if hparams.verbose:
    print('Measuring spike metrics...')

  start = time()

  fake_filename = utils.get_fake_filename(hparams, epoch)
  rearrange_saved_signals(hparams, fake_filename)

  metrics = populate_metrics_dict(hparams.num_processors, hparams.num_neurons)

  if hparams.num_processors > 1:
    pool = Pool(processes=hparams.num_processors)

    pool.starmap(
        neuron_spike_metrics,
        [(hparams, epoch, n, metrics) for n in range(hparams.num_neurons)])
    pool.close()
  else:
    for n in tqdm(
        range(hparams.num_neurons),
        desc='\tNeuron',
        disable=not bool(hparams.verbose)):
      neuron_spike_metrics(hparams, epoch, n, metrics)

  end = time()

  summary.scalar('elapse/spike_metrics', end - start, training=False)

  for key, value in metrics.items():
    if key.startswith('spike_metrics'):
      result = np.mean(value)
      if hparams.verbose:
        print('\t{}: {:.04f}'.format(key, result))
      summary.scalar(key, result, training=False)
    elif key.startswith('histogram'):
      for i, data in enumerate(value):
        summary.plot_histogram(
            '{}/neuron_{}'.format(key[key.find('/') + 1:], i),
            data,
            xlabel='Hz',
            ylabel='Amount',
            training=False)
