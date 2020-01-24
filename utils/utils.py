import os
import json
import pickle
import numpy as np
from time import time
from glob import glob
from multiprocessing import Pool

from .oasis_helper import deconvolve_signals
from .h5_helpers import open_h5, create_or_append_h5
from .metrics_helper import mean_spike_count, van_rossum_distance


def split(sequence, n):
  """ divide sequence into n sub-sequence evenly"""
  k, m = divmod(len(sequence), n)
  return [
      sequence[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)
  ]


def denormalize(x, x_min, x_max):
  """ re-scale signals back to its original range """
  return x * (x_max - x_min) + x_min


def store_hparams(hparams):
  with open(os.path.join(hparams.output_dir, 'hparams.json'), 'w') as file:
    json.dump(hparams.__dict__, file)


def get_signal_filename(hparams, epoch):
  """ return the filename of the signal h5 file given epoch """
  return os.path.join(hparams.output_dir,
                      'epoch{:03d}_signals.h5'.format(epoch))


def save_signals(hparams, epoch, real_signals, real_spikes, fake_signals):
  filename = get_signal_filename(hparams, epoch)

  real_signals = denormalize(
      real_signals, x_min=hparams.signals_min, x_max=hparams.signals_max)
  fake_signals = denormalize(
      fake_signals, x_min=hparams.signals_min, x_max=hparams.signals_max)

  with open_h5(filename, mode='a') as file:
    create_or_append_h5(file, 'real_spikes', real_spikes)
    create_or_append_h5(file, 'real_signals', real_signals)
    create_or_append_h5(file, 'fake_signals', fake_signals)


def save_models(hparams, generator, discriminator, epoch):
  ckpt_dir = os.path.join(hparams.output_dir, 'checkpoints')
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
  filename = os.path.join(ckpt_dir, 'epoch-{:03d}.pkl'.format(epoch))

  generator_weights = generator.get_weights()
  discriminator_weights = discriminator.get_weights()

  with open(filename, 'wb') as file:
    pickle.dump({
        'epoch': epoch,
        'generator_weights': generator_weights,
        'discriminator_weights': discriminator_weights
    }, file)

  print('Saved model checkpoint to {}\n'.format(filename))


def load_models(hparams, generator, discriminator):
  ckpts = glob(os.path.join(hparams.output_dir, 'ckpt-*'))
  if ckpts:
    ckpts.sort()
    filename = ckpts[-1]
    with open(filename, 'rb') as file:
      ckpt = pickle.load(file)
    generator.set_weights(ckpt['generator_weights'])
    discriminator.set_weights(ckpt['discriminator_weights'])
    print('restore checkpoint {}'.format(filename))


def deconvolve_saved_signals(hparams, filename):
  start = time()
  with open_h5(filename, mode='a') as file:
    fake_signals = file['fake_signals'][:]
    fake_spikes = deconvolve_signals(
        fake_signals, num_processors=hparams.num_processors)
    file.create_dataset(
        'fake_spikes',
        dtype=np.float32,
        data=fake_spikes,
        chunks=True,
        maxshape=(None, fake_spikes.shape[1], fake_spikes.shape[2]))
  elapse = time() - start
  print('deconvolve {} signals in {:.2f}s'.format(len(fake_spikes), elapse))


def van_rossum_distance_loop(args):
  real_spikes, fake_spikes = args
  assert real_spikes.shape == fake_spikes.shape
  shape = real_spikes.shape
  distances = np.zeros((shape[0], shape[1]), dtype=np.float32)
  for i in range(shape[0]):
    for neuron in range(shape[1]):
      distances[i][neuron] = van_rossum_distance(real_spikes[i][neuron],
                                                 fake_spikes[i][neuron])
  return distances


def get_mean_van_rossum_distance(hparams, real_spikes, fake_spikes):
  start = time()
  if hparams.num_processors > 2:
    num_jobs = min(len(real_spikes), hparams.num_processors)
    real_spikes_split = split(real_spikes, n=num_jobs)
    fake_spikes_split = split(fake_spikes, n=num_jobs)
    pool = Pool(processes=num_jobs)
    distances = pool.map(van_rossum_distance_loop,
                         list(zip(real_spikes_split, fake_spikes_split)))
    pool.close()
    distances = np.concatenate(distances, axis=0)
  else:
    distances = van_rossum_distance_loop((real_spikes, fake_spikes))
  mean_distance = np.mean(distances)
  elapse = time() - start
  print('mean van Rossum distance in {:.2f}s'.format(elapse))
  return mean_distance


def get_mean_spike_error(real_spikes, fake_spikes):
  real_mean_spike = mean_spike_count(real_spikes)
  fake_mean_spike = mean_spike_count(fake_spikes)
  return np.mean(np.square(real_mean_spike - fake_mean_spike))


def measure_spike_metrics(hparams, epoch, summary):
  filename = get_signal_filename(hparams, epoch)
  deconvolve_saved_signals(hparams, filename)

  with open_h5(filename, mode='r') as file:
    real_spikes = file['real_spikes'][:]
    fake_spikes = file['fake_spikes'][:]

  mean_spike_error = get_mean_spike_error(real_spikes, fake_spikes)
  van_rossum_distance = get_mean_van_rossum_distance(hparams, real_spikes,
                                                     fake_spikes)

  summary.scalar('spike_count_mse', mean_spike_error, training=False)
  summary.scalar('van_rossum_distance', van_rossum_distance, training=False)


def delete_generated_file(hparams, epoch):
  filename = get_signal_filename(hparams, epoch)
  if os.path.exists(filename):
    os.remove(filename)
