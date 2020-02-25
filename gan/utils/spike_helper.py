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


def rearrange_saved_signals(hparams, filename):
  ''' rearrange signals to (neurons, samples, segments) '''
  shape = (hparams.validation_size, hparams.num_neurons)
  with h5_helper.open_h5(filename, mode='r+') as file:
    for key in file.keys():
      value = file[key][:]
      # check if value has shape (samples, neurons)
      if value.shape[:2] == shape:
        value = utils.swap_neuron_major(hparams, value)
        h5_helper.overwrite_dataset(file, key, value)


def neuron_spike_metrics(hparams, epoch, neuron, metrics):
  """ measure spike metrics for neuron in file and write results to metrics """
  # get real neuron data
  real_filename = utils.get_real_neuron_filename(hparams, neuron)
  with open(real_filename, 'rb') as file:
    real_data = pickle.load(file)
  real_spikes = real_data['real_spikes']
  assert type(real_spikes) == list and type(real_spikes[0]) == SpikeTrain

  # get fake neuron data
  fake_filename = utils.get_fake_filename(hparams, epoch)
  with h5_helper.open_h5(fake_filename, mode='r') as file:
    fake_signals = file['fake_signals'][neuron]
  fake_spikes = deconvolve_signals(fake_signals, to_neo=True)

  assert len(real_spikes) == len(fake_spikes)

  if 'spike_metrics/firing_rate_error' in metrics:
    if 'firing_rate' in real_data:
      real_firing_rate = real_data['firing_rate']
    else:
      real_firing_rate = spike_metrics.mean_firing_rate(real_spikes)
      # cache firing rate data for neuron
      real_data['firing_rate'] = real_firing_rate
      with open(real_filename, 'wb') as file:
        pickle.dump(real_data, file)

    fake_firing_rate = spike_metrics.mean_firing_rate(fake_spikes)
    firing_rate_error = np.mean(np.square(real_firing_rate - fake_firing_rate))
    metrics['spike_metrics/firing_rate_error'][neuron] = firing_rate_error

    if 'histogram/firing_rate' in metrics:
      metrics['histogram/firing_rate'][neuron] = (real_firing_rate,
                                                  fake_firing_rate)
  if 'spike_metrics/cross_coefficient' in metrics:
    corrcoef = spike_metrics.correlation_coefficients(real_spikes, fake_spikes)
    metrics['spike_metrics/cross_coefficient'][neuron] = np.mean(corrcoef)

  if 'spike_metrics/covariance' in metrics:
    covariance = spike_metrics.covariance(real_spikes, fake_spikes)
    metrics['spike_metrics/covariance'][neuron] = np.mean(covariance)

  if 'spike_metrics/van_rossum_distance' in metrics:
    # compares to first 1000 samples to save time
    distance = spike_metrics.van_rossum_distance(real_spikes[:1000],
                                                 fake_spikes[:1000])
    metrics['spike_metrics/van_rossum_distance'][neuron] = np.mean(distance)

    if 'histogram/van_rossum_distance' in metrics:
      metrics['histogram/van_rossum_distance'][neuron] = (distance, [])

    if 'heatmap/van_rossum_distance' in metrics:
      metrics['heatmap/van_rossum_distance'][neuron] = distance


def populate_metrics_dict(num_processors, num_neurons):
  ''' create thread-safe dictionary to store metrics '''
  keys = [
      'spike_metrics/firing_rate_error',
      'histogram/firing_rate',
      'spike_metrics/covariance',
      'spike_metrics/van_rossum_distance',
      'histogram/van_rossum_distance',
      'heatmap/van_rossum_distance',
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
    tag = key[key.find('/') + 1:]
    if key.startswith('spike_metrics'):
      result = np.mean(value)
      if hparams.verbose:
        print('\t{}: {:.04f}'.format(key, result))
      summary.scalar(key, result, training=False)
    elif key.startswith('histogram'):
      for i, data in enumerate(value):
        summary.plot_histogram(
            '{}_histogram/neuron_{}'.format(tag, i),
            data,
            xlabel='Hz' if tag == 'firing_rate' else 'distance',
            ylabel='Amount',
            training=False)
    elif key.startswith('heatmap'):
      for i, data in enumerate(value):
        summary.plot_heatmap(
            '{}_heatmap/neuron_{}'.format(tag, i), data, training=False)
