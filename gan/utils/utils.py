import os
import json
import pickle
from glob import glob
import tensorflow as tf

from . import h5_helper


def split_index(length, n):
  """ return a list of (start, end) that divide length into n chunks """
  k, m = divmod(length, n)
  return [(i * k + min(i, m), (i + 1) * k + min(i + 1, m)) for i in range(n)]


def split(sequence, n):
  """ divide sequence into n sub-sequences evenly"""
  indexes = split_index(len(sequence), n)
  return [sequence[indexes[i][0]:indexes[i][1]] for i in range(len(indexes))]


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

  with h5_helper.open_h5(filename, mode='a') as file:
    h5_helper.create_or_append_h5(file, 'real_spikes', real_spikes)
    h5_helper.create_or_append_h5(file, 'real_signals', real_signals)
    h5_helper.create_or_append_h5(file, 'fake_signals', fake_signals)


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


def add_to_dict(dictionary, tag, value):
  """ Add tag with value to dictionary """
  if tag not in dictionary:
    dictionary[tag] = []
  dictionary[tag] += value if type(value) == list else [value]
