import os
import pickle
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm


def process_file(hparams, filename):
  print('processing file {}...'.format(filename))
  with open(filename, 'rb') as file:
    data = pickle.load(file)

  raw_signals = np.array(data['signals'], dtype=np.float32)
  raw_spikes = np.array(data['spikes'], dtype=np.float32)

  # remove first two rows in signals
  raw_signals = raw_signals[2:]
  raw_spikes = raw_spikes[2:]

  assert raw_signals.shape == raw_spikes.shape

  signals, spikes = [], []
  for i in tqdm(range(raw_signals.shape[-1] - hparams.sequence_length)):
    signals.append(raw_signals[:, i:i + hparams.sequence_length])
    spikes.append(raw_spikes[:, i:i + hparams.sequence_length])

  signals = np.concatenate(signals, axis=0)
  spikes = np.concatenate(spikes, axis=0)

  assert signals.shape == spikes.shape

  return signals, spikes


def main(hparams):
  if not os.path.exists(hparams.input_dir):
    print('input directory {} does not exists'.format(hparams.input_dir))
  if os.path.exists(hparams.output):
    print('output {} already exists\n'.format(hparams.output))

  filenames = glob(os.path.join(hparams.input_dir, '*.pkl'))
  filenames.sort()

  signals, spikes = [], []
  for filename in filenames:
    signal, spike = process_file(hparams, filename)
    signals.append(signal)
    spikes.append(spike)

  signals = np.concatenate(signals, axis=0)
  spikes = np.concatenate(spikes, axis=0)

  assert signals.shape == spikes.shape

  with open(hparams.output, 'wb') as file:
    pickle.dump({'signals': signals, 'spikes': spikes}, file)

  print('saved {} segments to {}'.format(len(signals), hparams.output))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', default='raw_data', type=str)
  parser.add_argument('--output', default='dataset.pkl', type=str)
  parser.add_argument('--sequence_length', default=120, type=int)
  hparams = parser.parse_args()
  main(hparams)
