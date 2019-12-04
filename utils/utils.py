import os
import json
import numpy as np
from time import time
from multiprocessing import Pool, cpu_count

from .oasis_helper import deconvolve_signals
from .h5_helpers import open_h5, create_or_append_h5
from .metrics_helper import mean_spike_count, van_rossum_distance


def split(sequence, n):
  """ divide sequence into n sub-sequence evenly"""
  k, m = divmod(len(sequence), n)
  return [
      sequence[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)
  ]


def store_hparams(hparams):
  with open(os.path.join(hparams.output_dir, 'hparams.json'), 'w') as file:
    json.dump(hparams.__dict__, file)


def get_signal_filename(hparams, epoch):
  """ return the filename of the signal h5 file given epoch """
  return os.path.join(hparams.output_dir,
                      'epoch{:03d}_signals.h5'.format(epoch))


def save_signals(hparams, epoch, real_spikes, real_signals, fake_signals):
  filename = get_signal_filename(hparams, epoch)

  with open_h5(filename, mode='a') as file:
    create_or_append_h5(file, 'real_spikes', real_spikes)
    create_or_append_h5(file, 'real_signals', real_signals)
    create_or_append_h5(file, 'fake_signals', fake_signals)


def deconvolve_saved_signals(hparams, epoch):
  start = time()
  filename = get_signal_filename(hparams, epoch)

  with open_h5(filename, mode='a') as file:
    fake_signals = file['fake_signals'][:]
    fake_spikes = deconvolve_signals(fake_signals, multiprocessing=True)
    file.create_dataset(
        'fake_spikes',
        dtype=np.float32,
        data=fake_spikes,
        chunks=True,
        maxshape=(None, fake_spikes.shape[1]))
  elapse = time() - start
  print('deconvolve {} signals in {:.2f}s'.format(len(fake_spikes), elapse))


def get_mean_spike_error(hparams, epoch):
  filename = get_signal_filename(hparams, epoch)
  with open_h5(filename, mode='r') as file:
    real_spikes = file['real_spikes'][:]
    fake_spikes = file['fake_spikes'][:]

  real_mean_spike = mean_spike_count(real_spikes)
  fake_mean_spike = mean_spike_count(fake_spikes)
  return real_mean_spike - fake_mean_spike


def _van_rossum_distance_loop(args):
  real_spikes, fake_spikes = args
  distances = []
  for i in range(len(real_spikes)):
    distances.append(van_rossum_distance(real_spikes[i], fake_spikes[i]))
  return np.array(distances, dtype=np.float32)


def get_mean_van_rossum_distance(hparams, epoch):
  start = time()
  filename = get_signal_filename(hparams, epoch)
  with open_h5(filename, mode='r') as file:
    real_spikes = file['real_spikes'][:]
    fake_spikes = file['fake_spikes'][:]

  num_jobs = min(len(real_spikes), cpu_count() - 2)
  real_spikes_split = split(real_spikes, n=num_jobs)
  fake_spikes_split = split(fake_spikes, n=num_jobs)
  pool = Pool(processes=num_jobs)
  distances = pool.map(_van_rossum_distance_loop,
                       list(zip(real_spikes_split, fake_spikes_split)))
  pool.close()
  distances = np.concatenate(distances, axis=0)
  mean_distiance = np.mean(distances)
  elapse = time() - start
  print('mean van Rossum distance in {:.2f}s'.format(elapse))
  return mean_distiance
