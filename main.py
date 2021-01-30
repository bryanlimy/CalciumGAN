import os
import argparse
import numpy as np
from time import time
from tqdm import tqdm
import tensorflow as tf
from shutil import rmtree
from tensorflow.keras import mixed_precision

from calciumgan.utils import utils
from calciumgan.models.registry import get_models
from calciumgan.utils.summary_helper import Summary
from calciumgan.utils.dataset_helper import get_dataset
from calciumgan.algorithms.registry import get_algorithm

np.random.seed(1234)
tf.random.set_seed(1234)


def train(hparams, ds, gan, summary, epoch):
  results = {}
  for signals in tqdm(
      ds,
      desc='Train',
      total=hparams.train_steps,
      disable=not bool(hparams.verbose)):
    result = gan.train(signals)
    hparams.global_step += 1
    utils.update_dict(results, result)
  for key, value in results.items():
    summary.scalar(f'loss/{key}', np.mean(value), epoch, training=True)


def validate(hparams, ds, gan, summary, epoch):
  results = {}
  for signals in tqdm(
      ds,
      desc='Validate',
      total=hparams.validation_steps,
      disable=not bool(hparams.verbose)):
    result = gan.validate(signals)
    utils.update_dict(results, result)
  for key, value in results.items():
    results[key] = np.mean(value)
    summary.scalar(f'loss/{key}', results[key], epoch, training=False)
  return results


def main(hparams):
  if hparams.clear_output_dir and os.path.exists(hparams.output_dir):
    rmtree(hparams.output_dir)

  tf.keras.backend.clear_session()

  if hparams.mixed_precision:
    mixed_precision.set_global_policy('mixed_float16')

  summary = Summary(hparams)
  train_ds, validation_ds = get_dataset(hparams, summary)
  G, D = get_models(hparams, summary)
  gan = get_algorithm(hparams, G, D)

  utils.save_hparams(hparams)

  # noise to test generator and plot to TensorBoard
  test_noise = gan.sample_noise(batch_size=1)

  for epoch in range(hparams.epochs):
    if hparams.verbose:
      print(f'Epoch {epoch + 1:03d}/{hparams.epochs:03d}')

    start = time()
    # train(hparams, train_ds, gan, summary, epoch)
    results = validate(hparams, validation_ds, gan, summary, epoch)
    end = time()

    summary.scalar('elapse', end - start, epoch, training=False)

    if hparams.verbose:
      print(f'G loss: {results["G_loss"]:.04f}\t'
            f'D loss: {results["D_loss"]:.04f}\n'
            f'Elapse: {(end - start)/60:.02f} mins\n')

    if epoch % 10 == 0 or epoch == hparams.epochs - 1:
      utils.plot_samples(
          hparams, summary, gan.generate(test_noise), epoch, tag='fake_signals')

  utils.save_models(hparams, gan)
  utils.save_samples(hparams, validation_ds, gan)
  utils.deconvolve_samples(hparams)

  # generate dataset for surrogate metrics
  if hparams.surrogate_ds:
    utils.generate_dataset(hparams, gan=gan, num_samples=2 * 10**6)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', default='dataset/tfrecords')
  parser.add_argument('--output_dir', default='runs')
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--num_units', default=32, type=int)
  parser.add_argument('--kernel_size', default=24, type=int)
  parser.add_argument('--strides', default=2, type=int)
  parser.add_argument('--m', default=2, type=int, help='phase shuffle m')
  parser.add_argument('--n', default=2, type=int, help='phase shuffle n')
  parser.add_argument('--epochs', default=20, type=int)
  parser.add_argument('--dropout', default=0.2, type=float)
  parser.add_argument('--learning_rate', default=0.0001, type=float)
  parser.add_argument('--noise_dim', default=32, type=int)
  parser.add_argument('--gradient_penalty', default=10.0, type=float)
  parser.add_argument('--model', default='calciumgan', type=str)
  parser.add_argument('--activation', default='leakyrelu', type=str)
  parser.add_argument('--batch_norm', action='store_true')
  parser.add_argument('--layer_norm', action='store_true')
  parser.add_argument('--algorithm', default='gan', type=str)
  parser.add_argument(
      '--n_critic',
      default=5,
      type=int,
      help='number of steps between each generator update')
  parser.add_argument('--clear_output_dir', action='store_true')
  parser.add_argument('--plot_weights', action='store_true')
  parser.add_argument('--mixed_precision', action='store_true')
  parser.add_argument(
      '--profile', action='store_true', help='enable TensorBoard profiling')
  parser.add_argument('--dpi', default=120, type=int)
  parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2])
  params = parser.parse_args()

  params.global_step = 0
  params.surrogate_ds = True if 'surrogate' in params.input_dir else False
  main(params)
