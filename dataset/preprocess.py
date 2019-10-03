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

  signals = np.array(data['signals'], dtype=np.float32)
  time = np.array(data['time'], dtype=np.float32)
  position = np.array(data['position'], dtype=np.float32)

  # remove first two rows in signals
  signals = signals[2:]

  segments = []
  for i in tqdm(range(signals.shape[-1] - hparams.sequence_length)):
    segment = list(signals[:, i:i + hparams.sequence_length])
    segments.extend(segment)

  return segments


def main(hparams):
  if not os.path.exists(hparams.input_dir):
    print('input directory {} does not exists'.format(hparams.input_dir))
  if os.path.exists(hparams.output):
    print('output {} already exists'.format(hparams.output))

  filenames = glob(os.path.join(hparams.input_dir, '*.pkl'))
  filenames.sort()

  segments = []
  for filename in filenames:
    segments += process_file(hparams, filename)

  segments = np.array(segments, dtype=np.float32)

  with open(hparams.output, 'wb') as file:
    pickle.dump(segments, file)

  print('saved {} segments to {}'.format(len(segments), hparams.output))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', default='raw_data', type=str)
  parser.add_argument('--output', default='dataset.pkl', type=str)
  parser.add_argument('--sequence_length', default=40, type=int)
  hparams = parser.parse_args()
  main(hparams)
