import os
import json
import pickle
import numpy as np
from time import time
from glob import glob
import tensorflow as tf
from shutil import rmtree
from multiprocessing import Pool

from . import h5_helpers
from . import spike_helper
from . import spike_metrics


def split(sequence, n):
  """ divide sequence into n sub-sequence evenly"""
  k, m = divmod(len(sequence), n)
  return [
      sequence[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)
  ]


def normalize(x, x_min, x_max):
  # scale x to be between 0 and 1
  return (x - x_min) / (x_max - x_min)


def denormalize(x, x_min, x_max):
  ''' re-scale signals back to its original range '''
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

  if hparams.normalize:
    real_signals = denormalize(
        real_signals, x_min=hparams.signals_min, x_max=hparams.signals_max)
    fake_signals = denormalize(
        fake_signals, x_min=hparams.signals_min, x_max=hparams.signals_max)

  with h5_helpers.open_h5(filename, mode='a') as file:
    h5_helpers.create_or_append_h5(file, 'real_spikes', real_spikes)
    h5_helpers.create_or_append_h5(file, 'real_signals', real_signals)
    h5_helpers.create_or_append_h5(file, 'fake_signals', fake_signals)


def delete_saved_signals(hparams, epoch):
  filename = get_signal_filename(hparams, epoch)
  if os.path.exists(filename):
    os.remove(filename)


def save_models(hparams, gan, epoch):
  ckpt_dir = os.path.join(hparams.output_dir, 'checkpoints')
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
  filename = os.path.join(ckpt_dir, 'epoch-{:03d}.pkl'.format(epoch))

  with open(filename, 'wb') as file:
    pickle.dump({
        'epoch': epoch,
        'generator_weights': gan.generator.get_weights(),
        'discriminator_weights': gan.discriminator.get_weights()
    }, file)

  if hparams.verbose:
    print('Saved model checkpoint to {}\n'.format(filename))


def load_models(hparams, generator, discriminator):
  ckpts = glob(os.path.join(hparams.output_dir, 'epoch-*'))
  if ckpts:
    ckpts.sort()
    filename = ckpts[-1]
    with open(filename, 'rb') as file:
      ckpt = pickle.load(file)
    generator.set_weights(ckpt['generator_weights'])
    discriminator.set_weights(ckpt['discriminator_weights'])

    if hparams.verbose:
      print('restore checkpoint {}'.format(filename))


def preform_spike_metrics(hparams, epoch):
  ''' Return True if need to preform spike metrics '''
  return hparams.spike_metrics and (epoch % hparams.spike_metrics_freq == 0 or
                                    epoch == hparams.epochs - 1)


def deconvolve_saved_signals(hparams, filename):
  start = time()
  with h5_helpers.open_h5(filename, mode='a') as file:
    fake_signals = file['fake_signals'][:50]
    fake_spikes = spike_helper.deconvolve_signals(
        fake_signals, num_processors=hparams.num_processors)

    file.create_dataset(
        'fake_spikes',
        dtype=fake_spikes.dtype,
        data=fake_spikes,
        chunks=True,
        maxshape=(None, fake_spikes.shape[1]))
  elapse = time() - start

  if hparams.verbose:
    print('Deconvolve {} signals in {:.2f}s'.format(len(fake_spikes), elapse))


def compute_spike_metrics(hparams, epoch, summary):
  filename = get_signal_filename(hparams, epoch)
  deconvolve_saved_signals(hparams, filename)

  with h5_helpers.open_h5(filename, mode='r') as file:
    real_spikes = file['real_spikes'][:]
    fake_spikes = file['fake_spikes'][:]

  # assert real_spikes.shape == fake_spikes.shape

  real_spikes = spike_helper.numpy_to_neo(real_spikes)
  fake_spikes = spike_helper.numpy_to_neo(fake_spikes)

  # spike_count_error = spike_metrics.mean_spike_count_error(
  #     real_spikes, fake_spikes)
  # firing_rate_error = spike_metrics.mean_firing_rate_error(
  #     real_spikes, fake_spikes)
  van_rossum_distance = spike_metrics.van_rossum_error(
      real_spikes, fake_spikes, num_processors=hparams.num_processors)

  # summary.scalar(
  #     'spike_metrics/spike_count_error', spike_count_error, training=False)
  # summary.scalar(
  #     'spike_metrics/van_rossum_distance', van_rossum_distance, training=False)
