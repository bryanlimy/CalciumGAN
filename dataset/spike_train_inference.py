import os
import pickle
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from oasis.functions import deconvolve
from oasis.oasis_methods import oasisAR1


def generate_spike_train(hparams, filename):
  print('processing file {}...'.format(filename))

  with open(filename, 'rb') as file:
    data = pickle.load(file)

  if 'oasis' in data:
    print('oasis spike train already existed in {}'.format(filename))
    if hparams.overwrite:
      print('overwriting...')
    else:
      return

  oasis = np.zeros((data['signals'].shape), dtype=data['signals'].dtype)
  for i in tqdm(range(len(data['signals']))):
    _, oasis[i] = oasisAR1(data['signals'][i], g=0.95, s_min=.55)
  oasis = np.where(oasis > 0.5, 1.0, 0.0)

  data['oasis'] = np.array(oasis, dtype=np.float32)

  with open(filename, 'wb') as file:
    pickle.dump(data, file)


def remove_oasis(filename):
  print('cleaning file {}...'.format(filename))
  with open(filename, 'rb') as file:
    data = pickle.load(file)

  if 'oasis' in data:
    del data['oasis']
    with open(filename, 'wb') as file:
      pickle.dump(data, file)


def main(hparams):
  filenames = glob(os.path.join(hparams.input_dir, '*.pkl'))
  filenames.sort()

  for filename in filenames:
    if hparams.clean:
      remove_oasis(filename)
    else:
      generate_spike_train(hparams, filename)

  print('process completed')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', default='raw_data', type=str)
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--clean', action='store_true')
  hparams = parser.parse_args()
  main(hparams)
