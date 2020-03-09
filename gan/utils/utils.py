import os
import json
import pickle
import subprocess
import numpy as np
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
  ''' scale x to be between 0 and 1 '''
  return (x - x_min) / (x_max - x_min)


def denormalize(x, x_min, x_max):
  ''' re-scale signals back to its original range '''
  return x * (x_max - x_min) + x_min


def get_current_git_hash():
  ''' return the current Git hash '''
  return subprocess.check_output(['git', 'describe',
                                  '--always']).strip().decode()


def store_hparams(hparams):
  hparams.git_hash = get_current_git_hash()
  with open(os.path.join(hparams.output_dir, 'hparams.json'), 'w') as file:
    json.dump(hparams.__dict__, file)


def swap_neuron_major(hparams, array):
  shape = (hparams.validation_size, hparams.num_neurons)
  return np.swapaxes(
      array, axis1=0, axis2=1) if array.shape[:2] == shape else array


def save_fake_signals(hparams, epoch, fake_signals):
  if hparams.normalize:
    fake_signals = denormalize(
        fake_signals, x_min=hparams.signals_min, x_max=hparams.signals_max)

  filename = os.path.join(hparams.generated_dir,
                          'epoch{:03d}_signals.h5'.format(epoch))

  h5_helper.write(filename, {'signals': fake_signals})

  # store generated data information
  info_filename = os.path.join(hparams.generated_dir, 'info.pkl')
  info = {}
  if os.path.exists(info_filename):
    with open(info_filename, 'rb') as file:
      info = pickle.load(file)
  info[epoch] = {'global_step': hparams.global_step, 'filename': filename}
  with open(info_filename, 'wb') as file:
    pickle.dump(info, file)


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
    print('Saved checkpoint to {}\n'.format(filename))


def load_models(hparams, generator, discriminator):
  ckpts = glob(os.path.join(hparams.output_dir, 'checkpoints', 'epoch-*'))
  if ckpts:
    ckpts.sort()
    filename = ckpts[-1]
    with open(filename, 'rb') as file:
      ckpt = pickle.load(file)
    generator.set_weights(ckpt['generator_weights'])
    discriminator.set_weights(ckpt['discriminator_weights'])
    if hparams.verbose:
      print('Restored checkpoint at {}'.format(filename))


def is_neuron_major(array, hparams):
  ''' return True if the array is neuron-major '''
  # TODO make it works with 3 dimensional array
  return array.shape[0] == hparams.num_neurons


def set_array_format(array, format, hparams):
  # TODO make it works with 3 dimensional array
  assert len(array.shape) == len(format)

  # get NWC format index
  shape = list(array.shape)
  newshape = [
      shape.index(hparams.sequence_length if i == 'W' else hparams.num_neurons)
      for i in format
  ]

  if sorted(newshape) == newshape:
    return array
  elif type(array) == np.ndarray:
    return np.swapaxes(array, newshape[0], newshape[1])
  else:
    return tf.transpose(array, perm=newshape)
