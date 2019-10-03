import argparse
import tensorflow as tf


def main(hparams):
  pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', default='dataset/dataset.pkl')
  parser.add_argument('--output_dir', default='runs')
  parser.add_argument('--batch_size', default=128, type=int)
  parser.add_argument('--epochs', default=20, type=int)
  parser.add_argument('--num_units', default=256, type=int)
  parser.add_argument('--dropout', default=0.2, type=float)
  parser.add_argument('--lr', default=0.001, type=float)
  parser.add_argument('--model', default='mlp')
  parser.add_argument('--activation', default='relu')
  parser.add_argument('--verbose', default=1, type=int)
  hparams = parser.parse_args()
  main(hparams)
