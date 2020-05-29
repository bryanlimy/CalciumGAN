import os
import pickle
import warnings
import platform
import argparse
import numpy as np
from tqdm import tqdm

from gan.utils import utils

# import matplotlib
#
# if platform.system() == 'Darwin':
#   matplotlib.use('TkAgg')
#
# import matplotlib.pyplot as plt
# plt.style.use('seaborn-deep')
#
# import seaborn as sns


def load_info(hparams):
  filename = os.path.join(hparams.generated_dir, 'info.pkl')
  with open(filename, 'rb') as file:
    info = pickle.load(file)
  return info


def get_pickle_data(filename, name='spikes'):
  with open(filename, 'rb') as file:
    data = pickle.load(file)
  return data[name]


from gan.utils import spike_helper


def get_generate_data(hparams):
  filename = os.path.join(hparams.output_dir, 'generated.pkl')
  if not os.path.exists(filename):
    print('generated pickle file not found in {}'.format(hparams.output_dir))
    exit()

  with open(filename, 'rb') as file:
    data = pickle.load(file)

  if 'spikes' in data:
    spikes = data['spikes']
  else:
    signals = utils.set_array_format(
        data['signals'], data_format='NCW', hparams=hparams)
    spikes = np.zeros(signals.shape, dtype=np.float32)
    for i in tqdm(range(len(signals)), desc='Deconvolution'):
      spikes[i] = spike_helper.deconvolve_signals(signals[i], threshold=0.5)
    with open(filename, 'wb') as file:
      pickle.dump({'signals': signals, 'spikes': spikes}, file)

  return utils.set_array_format(spikes, data_format='NCW', hparams=hparams)


def main(hparams):
  if not os.path.exists(hparams.output_dir):
    print('{} not found'.format(hparams.output_dir))
    exit()

  utils.load_hparams(hparams)

  ground_truth_data = get_pickle_data(
      os.path.join(hparams.input_dir, 'ground_truth.pkl'))
  surrogate_data = get_pickle_data(
      os.path.join(hparams.input_dir, 'surrogate.pkl'))
  generated_data = get_generate_data(hparams)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', default='runs', type=str)
  parser.add_argument('--num_processors', default=4, type=int)
  hparams = parser.parse_args()

  warnings.simplefilter(action='ignore', category=UserWarning)
  warnings.simplefilter(action='ignore', category=RuntimeWarning)
  warnings.simplefilter(action='ignore', category=DeprecationWarning)

  main(hparams)
