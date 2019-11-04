import os
import pickle
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from oasis.functions import deconvolve


def generate_spike_train(hparams, filename):
  print('processing file {}...'.format(filename))

  with open(filename, 'rb') as file:
    data = pickle.load(file)

  if 'oasis' in data:
    print('oasis spike train already existed in {}'.format(filename))
    return

  results = []
  for signal in tqdm(data['signals']):
    c, s, b, g, lam = deconvolve(signal, g=(None, None), penalty=1)
    results.append(s / s.max())

  data['oasis'] = np.array(results, dtype=np.float32)

  with open(filename, 'wb') as file:
    pickle.dump(data, file)


def main(hparams):
  filenames = glob(os.path.join(hparams.input_dir, '*.pkl'))
  filenames.sort()

  for filename in filenames:
    generate_spike_train(hparams, filename)

  print('deconvolution completed')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', default='raw_data', type=str)
  hparams = parser.parse_args()
  main(hparams)
