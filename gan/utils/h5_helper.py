import os
import h5py
import numpy as np

from . import utils


def append(ds, value):
  """ append value to a H5 dataset """
  if type(value) != np.ndarray:
    value = np.array(value, dtype=np.float32)
  ds.resize((ds.shape[0] + value.shape[0]), axis=0)
  ds[-value.shape[0]:] = value


def write(filename, content):
  """ create dataset and write content to H5 file
  NOTE: dataset must be stored in NWC data format
  """
  assert type(content) == dict
  with h5py.File(filename, mode='a') as file:
    for k, v in content.items():
      file.create_dataset(k, shape=v.shape, dtype=v.dtype, data=v)


def overwrite(filename, name, value):
  ''' overwrite dataset with value '''
  with h5py.File(filename, mode='r+') as file:
    if name not in file.keys():
      raise KeyError('{} cannot be found'.format(name))
    del file[name]
    file.create_dataset(name, shape=value.shape, dtype=np.float32, data=value)


def get(filename, name, neuron=None, sample=None):
  """
  Return the dataset with the given name
  NOTE: Dataset must be stored in NWC format
  neuron: index of the specific neuron to be returned
  sample: index of the specific sample to be returned
  """
  assert not (neuron is not None and sample is not None)
  with h5py.File(filename, mode='r') as file:
    if name not in file.keys():
      raise KeyError('{} cannot be found'.format(name))
    ds = file[name]
    if neuron is not None:
      return ds[:, :, neuron]
    elif sample is not None:
      return ds[sample, :, :]
    else:
      return ds[:]


def get_dataset_length(filename, name):
  with h5py.File(filename, mode='r') as file:
    dataset = file[name]
    length = dataset.len()
  return length


def contains(filename, name):
  with h5py.File(filename, mode='r') as file:
    keys = list(file.keys())
  return name in keys
