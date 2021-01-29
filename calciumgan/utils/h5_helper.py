import h5py


def append(ds, value):
  """ append value to a H5 dataset """
  ds.resize((ds.shape[0] + value.shape[0]), axis=0)
  ds[-value.shape[0]:] = value


def write(filename, data):
  """ write or append content to H5 file
  NOTE: dataset must be stored in NWC data format
  """
  assert type(data) == dict
  with h5py.File(filename, mode='a') as file:
    for key, value in data.items():
      if key in file.keys():
        del file[key]
      file.create_dataset(key, shape=value.shape, dtype=value.dtype, data=value)


def overwrite(filename, key, value):
  ''' overwrite dataset with value '''
  with h5py.File(filename, mode='r+') as file:
    if key not in file.keys():
      raise KeyError('{} cannot be found'.format(key))
    del file[key]
    file.create_dataset(key, shape=value.shape, dtype=value.dtype, data=value)


def get(filename, key, neuron=None, trial=None):
  """
  Return the dataset with the given name
  NOTE: Dataset must be stored in NWC format
  neuron: index of the specific neuron to be returned
  trial: index of the specific trial to be returned
  """
  assert not (neuron is not None and trial is not None)
  with h5py.File(filename, mode='r') as file:
    if key not in file.keys():
      raise KeyError('{} cannot be found'.format(key))
    ds = file[key]
    if neuron is not None:
      return ds[:, :, neuron]
    elif trial is not None:
      return ds[trial, :, :]
    else:
      return ds[:]


def get_keys(filename):
  with h5py.File(filename, mode='r') as file:
    keys = [k for k in file.keys()]
  return keys


def get_length(filename, key):
  with h5py.File(filename, mode='r') as file:
    dataset = file[key]
    length = dataset.len()
  return length


def contains(filename, key):
  with h5py.File(filename, mode='r') as file:
    keys = list(file.keys())
  return key in keys
