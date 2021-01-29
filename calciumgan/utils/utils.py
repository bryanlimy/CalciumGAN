import os
import json
import pickle
import subprocess
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from calciumgan.utils import h5_helper as h5


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


def fft(signals):
  """ Apply FFT over each neuron recordings """
  real = np.zeros(signals.shape, dtype=np.float32)
  imag = np.zeros(signals.shape, dtype=np.float32)

  for b in tqdm(range(signals.shape[0])):
    for n in range(signals.shape[-1]):
      x = signals[b, :, n]
      x = tf.signal.fft(x.astype(np.complex64))
      x = x.numpy()
      real[b, :, n], imag[b, :, n] = np.real(x), np.imag(x)

  return np.concatenate([real, imag], axis=-1)


def ifft(signals):
  # signals shape (batch size, sequence, num neurons * 2)
  mid = signals.shape[-1] // 2
  real, imag = signals[..., :mid], signals[..., mid:]
  result = np.zeros(real.shape, np.float32)
  for b in range(real.shape[0]):
    for n in range(real.shape[-1]):
      x = real[b, :, n] + imag[b, :, n] * 1j
      x = tf.signal.ifft(x)
      x = x.numpy()
      result[b, :, n] = np.real(x)
  return result


def reverse_preprocessing(hparams, x):
  ''' reverse the preprocessing on data so that it matches the input data '''
  if hparams.normalize:
    x = denormalize(x, x_min=hparams.signals_min, x_max=hparams.signals_max)

  if hparams.conv2d:
    if hparams.fft:
      x = np.concatenate((x[..., 0], x[..., 1]), axis=-1)
    else:
      x = np.squeeze(x, axis=-1)

  if hparams.fft:
    x = ifft(x)

  return x


def plot_samples(hparams, summary, signals, step=0, tag='traces'):
  signals = reverse_preprocessing(hparams, signals)
  signals = set_array_format(signals[0], data_format='CW', hparams=hparams)
  summary.plot_traces(
      tag, signals[hparams.focus_neurons], step=step, training=False)


def get_current_git_hash():
  ''' return the current Git hash '''
  return subprocess.check_output(['git', 'describe',
                                  '--always']).strip().decode()


def update_dict(target, source, replace=False):
  """ add or update items in source to target """
  for key, value in source.items():
    if replace:
      target[key] = value
    else:
      if key not in target:
        target[key] = []
      source[key].append(value)


def save_json(filename, data):
  assert type(data) == dict
  for key, value in data.items():
    if isinstance(value, np.ndarray):
      data[key] = value.tolist()
    elif isinstance(value, np.float32):
      data[key] = float(value)
  with open(filename, 'w') as file:
    json.dump(data, file)


def update_json(filename, data):
  content = {}
  if os.path.exists(filename):
    content = load_json(filename)
  for key, value in data.items():
    content[key] = value
  save_json(filename, content)


def load_json(filename):
  with open(filename, 'r') as file:
    content = json.load(file)
  return content


def save_hparams(hparams):
  hparams.hparams_filename = os.path.join(hparams.output_dir, 'hparams.json')
  hparams.git_hash = get_current_git_hash()
  with open(hparams.hparams_filename, 'w') as file:
    json.dump(hparams.__dict__, file)


def load_hparams(hparams):
  filename = os.path.join(hparams.output_dir, 'hparams.json')
  with open(filename, 'r') as file:
    content = json.load(file)
  for key, value in content.items():
    if not hasattr(hparams, key):
      setattr(hparams, key, value)


def swap_neuron_major(hparams, array):
  shape = (hparams.validation_size, hparams.num_neurons)
  return np.swapaxes(
      array, axis1=0, axis2=1) if array.shape[:2] == shape else array


def save_samples(hparams, ds, gan):
  if hparams.verbose:
    print('generating samples for evaluation...')

  samples = {'real': [], 'fake': []}
  for real in ds:
    noise = gan.sample_noise(real.shape[0])
    fake = gan.generate(noise, denorm=False)
    real = reverse_preprocessing(hparams, real)
    fake = reverse_preprocessing(hparams, fake)
    samples['real'].append(real.numpy())
    samples['fake'].append(fake.numpy())
  samples = {key: np.vstack(value) for key, value in samples.items()}

  if not os.path.exists(hparams.samples_dir):
    os.makedirs(hparams.samples_dir)
  hparams.signals_filename = os.path.join(hparams.samples_dir, 'signals.h5')
  h5.write(hparams.signals_filename, data=samples)
  update_json(
      filename=hparams.hparams_filename,
      data={
          'global_step': hparams.global_step,
          'signals_filename': hparams.signals_filename
      })
  if hparams.verbose:
    print(f'saved signal samples to {hparams.signals_filename}')


def save_models(hparams, gan):
  for model in gan.get_models():
    model.save_weights(os.path.join(hparams.checkpoint_dir, model.name))
  if hparams.verbose:
    print(f'checkpoints saved at {hparams.checkpoint_dir}/n')


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


def generate_dataset(hparams, gan, num_samples=1000):
  generated = np.zeros((num_samples,) + hparams.signal_shape, dtype=np.float32)
  batch_size = 100
  for i in tqdm(
      range(0, num_samples, batch_size),
      desc='Surrogate',
      disable=not bool(hparams.verbose)):
    noise = gan.get_noise(batch_size)
    signals = gan.generate(noise, denorm=True)
    generated[i:i + batch_size] = signals

  filename = os.path.join(hparams.output_dir, 'generated.pkl')
  with open(filename, 'wb') as file:
    pickle.dump({'signals': generated}, file)

  if hparams.verbose:
    print('save {} samples to {}'.format(num_samples, filename))
