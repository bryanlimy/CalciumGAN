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


def ifft(x):
  if tf.is_tensor(x):
    x = x.numpy()
  x = x[..., 0::2] + x[..., 1::2] * 1j
  x = np.fft.ifft(x, norm='ortho')
  return np.real(x)


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


def save_fake_signals(hparams, epoch, signals):
  if tf.is_tensor(signals):
    signals = signals.numpy()

  if hparams.normalize:
    signals = denormalize(
        signals, x_min=hparams.signals_min, x_max=hparams.signals_max)

  if hparams.fft:
    signals = ifft(signals)

  filename = os.path.join(hparams.generated_dir,
                          'epoch{:03d}_signals.h5'.format(epoch))

  h5_helper.write(filename, {'signals': signals.astype(np.float32)})

  # store generated data information
  info_filename = os.path.join(hparams.generated_dir, 'info.pkl')
  info = {}
  if os.path.exists(info_filename):
    with open(info_filename, 'rb') as file:
      info = pickle.load(file)
  if epoch not in info:
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
    print('Saved checkpoint to {}'.format(filename))


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


def get_array_format(shape, hparams):
  ''' get the array data format in string
  N: number of samples
  W: sequence length
  C: number of channels
  '''
  assert len(shape) <= 3
  return ''.join([
      'W' if s == hparams.sequence_length else
      'C' if s == hparams.num_neurons else 'N' for s in shape
  ])


def set_array_format(array, data_format, hparams):
  ''' set array to the given data format '''
  assert len(array.shape) == len(data_format)

  current_format = get_array_format(array.shape, hparams)

  assert set(current_format) == set(data_format)

  if data_format == current_format:
    return array

  perm = [current_format.index(s) for s in data_format]

  if tf.is_tensor(array):
    return tf.transpose(array, perm=perm)
  else:
    return np.transpose(array, axes=perm)


def remove_nan(array):
  return array[np.logical_not(np.isnan(array))]
