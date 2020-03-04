import os
import h5py
import numpy as np
from time import sleep

from . import utils


def _open(filename, mode):
  if mode == 'r' or mode == 'r+':
    if not os.path.exists(filename):
      print('{} not found'.format(filename))
      exit()
  while True:
    try:
      file = h5py.File(filename, mode=mode)
      break
    except OSError:
      sleep(1)
  return file


def append(ds, value):
  """ append value to a H5 dataset """
  if type(value) != np.ndarray:
    value = np.array(value, dtype=np.float32)
  ds.resize((ds.shape[0] + value.shape[0]), axis=0)
  ds[-value.shape[0]:] = value


def write(filename, content):
  """ create dataset and write content to H5 file """
  assert type(content) == dict
  with _open(filename, mode='a') as file:
    for key, value in content.items():
      file.create_dataset(key, shape=value.shape, dtype=np.float32, data=value)


def overwrite(filename, name, value):
  ''' overwrite dataset with value '''
  with _open(filename, mode='r+') as file:
    if name not in file.keys():
      raise KeyError('{} cannot be found'.format(name))
    del file[name]
    file.create_dataset(name, shape=value.shape, dtype=np.float32, data=value)


def get(filename, name, index=None, neuron=None, hparams=None):
  """
  Return the dataset with the given name and index
  If index is specified, neuron and hparams must also be provided
  :param filename: h5 filename
  :param name: name of the dataset
  :param index: (Optional) index in the dataset to return, otherwise return 
                            the whole dataset 
  :param neuron: (Optional) True if index is for a specific neuron
                            False if index is for a specific sample
  :param hparams: (Optional) hparams dict
  :return: dataset
  """
  with _open(filename, mode='r') as file:
    if name not in file.keys():
      raise KeyError('{} cannot be found'.format(name))
    ds = file[name]
    if index is None:
      return ds[:]
    else:
      assert type(neuron) is bool and hparams is not None
      if utils.is_neuron_major(ds, hparams):
        return ds[index, :, :] if neuron else ds[:, index, :]
      else:
        return ds[:, index, :] if neuron else ds[index, :, :]


def get_dataset_length(filename, name):
  with _open(filename, mode='r') as file:
    dataset = file[name]
    length = dataset.len()
  return length


def contains(filename, name):
  with _open(filename, mode='r') as file:
    keys = list(file.keys())
  return name in keys
