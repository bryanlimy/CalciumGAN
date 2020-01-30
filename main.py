import os
import argparse
import numpy as np
from time import time
from tqdm import tqdm
import tensorflow as tf
from shutil import rmtree

np.random.seed(1234)
tf.random.set_seed(1234)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from models.registry import get_models
from utils.summary_helper import Summary
from utils.dataset_helper import get_dataset
from algorithms.registry import get_algorithm
from utils.utils import store_hparams, save_signals, measure_spike_metrics, \
  save_models, load_models, delete_generated_file


def train(hparams, train_ds, gan, summary, epoch):
  gen_losses, dis_losses = [], []

  start = time()

  for signal, spike in tqdm(
      train_ds,
      desc='Epoch {:03d}/{:03d}'.format(epoch, hparams.epochs),
      total=hparams.steps_per_epoch,
      disable=not bool(hparams.verbose)):

    gen_loss, dis_loss, gradient_penalty, metrics = gan.train(signal)

    if hparams.global_step % hparams.summary_freq == 0:
      summary.log(
          gen_loss,
          dis_loss,
          gradient_penalty,
          metrics=metrics,
          gan=gan,
          training=True)

    gen_losses.append(gen_loss)
    dis_losses.append(dis_loss)

    hparams.global_step += 1

  end = time()

  summary.scalar('elapse (s)', end - start, step=epoch, training=True)

  return np.mean(gen_losses), np.mean(dis_losses)


def validate(hparams, validation_ds, gan, summary, epoch):
  gen_losses, dis_losses, gradient_penalties = [], [], []
  kl_divergences, mean_signals_errors, std_signals_errors = [], [], []
  min_signals_errors, max_signals_errors = [], []

  start = time()

  for signal, spike in validation_ds:
    fake, gen_loss, dis_loss, gradient_penalty, metrics = gan.validate(signal)

    gen_losses.append(gen_loss)
    dis_losses.append(dis_loss)
    if gradient_penalty is not None:
      gradient_penalties.append(gradient_penalty)
    kl_divergences.append(metrics['kl_divergence'])
    min_signals_errors.append(metrics['min_signals_error'])
    max_signals_errors.append(metrics['max_signals_error'])
    mean_signals_errors.append(metrics['mean_signals_error'])
    std_signals_errors.append(metrics['std_signals_error'])

    save_signals(
        hparams,
        epoch,
        real_signals=signal.numpy(),
        real_spikes=spike.numpy(),
        fake_signals=fake.numpy())

  end = time()

  gen_loss, dis_loss = np.mean(gen_losses), np.mean(dis_losses)
  gradient_penalty = np.mean(gradient_penalties) if gradient_penalties else None

  summary.log(
      gen_loss,
      dis_loss,
      gradient_penalty,
      metrics={
          'kl_divergence': np.mean(kl_divergences),
          'min_signals_error': np.mean(min_signals_errors),
          'max_signals_error': np.mean(max_signals_errors),
          'mean_signals_error': np.mean(mean_signals_errors),
          'std_signals_error': np.mean(std_signals_errors)
      },
      elapse=end - start,
      training=False)

  # evaluate spike metrics every 5 epochs
  if not hparams.skip_spike_metrics and (epoch % 5 == 0 or
                                         epoch == hparams.epochs - 1):
    measure_spike_metrics(hparams, epoch, summary)

  # delete generated signals and spike train
  if not hparams.keep_generated:
    delete_generated_file(hparams, epoch)

  return gen_loss, dis_loss


def train_and_validate(hparams, train_ds, validation_ds, gan, summary):
  # noise to test generator and plot to TensorBoard
  test_noise = gan.get_noise(batch_size=1)

  for epoch in range(hparams.epochs):

    train_gen_loss, train_dis_loss = train(
        hparams, train_ds, gan=gan, summary=summary, epoch=epoch)

    val_gen_loss, val_dis_loss = validate(
        hparams, validation_ds, gan=gan, summary=summary, epoch=epoch)

    # test generated data and plot in TensorBoard
    fake = gan.generate(test_noise)
    if hparams.input_dir == 'fashion_mnist':
      summary.image('fake', signals=fake, training=False)
    else:
      summary.plot_traces('fake', signals=fake, training=False)

    if hparams.verbose:
      print('Train: generator loss {:.4f} discriminator loss {:.4f}\n'
            'Eval: generator loss {:.4f} discriminator loss {:.4f}\n'.format(
                train_gen_loss, train_dis_loss, val_gen_loss, val_dis_loss))

    if not hparams.skip_checkpoint and (epoch % 5 == 0 or
                                        epoch == hparams.epochs - 1):
      save_models(hparams, gan, epoch)


def test(validation_ds, gan):
  gen_losses, dis_losses, gradient_penalties = [], [], []
  kl_divergences, mean_signals_errors, std_signals_errors = [], [], []

  for signal, spike in validation_ds:
    _, gen_loss, dis_loss, _, metrics = gan.validate(signal)

    gen_losses.append(gen_loss)
    dis_losses.append(dis_loss)
    kl_divergences.append(metrics['kl_divergence'])
    mean_signals_errors.append(metrics['mean_signals_error'])
    std_signals_errors.append(metrics['std_signals_error'])

  return {
      'kl_divergence': np.mean(kl_divergences),
      'mean_signals_error': np.mean(mean_signals_errors),
      'std_signals_error': np.mean(std_signals_errors)
  }


def main(hparams, return_metrics=False):
  if hparams.clear_output_dir and os.path.exists(hparams.output_dir):
    rmtree(hparams.output_dir)

  tf.keras.backend.clear_session()

  summary = Summary(hparams)

  train_ds, validation_ds = get_dataset(hparams, summary)

  generator, discriminator = get_models(hparams)

  if hparams.verbose:
    generator.summary()
    discriminator.summary()

  store_hparams(hparams)

  load_models(hparams, generator, discriminator)

  gan = get_algorithm(hparams, generator, discriminator, summary)

  train_and_validate(
      hparams,
      train_ds=train_ds,
      validation_ds=validation_ds,
      gan=gan,
      summary=summary)

  if return_metrics:
    return test(validation_ds, gan)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', default='dataset/tfrecords')
  parser.add_argument('--output_dir', default='runs')
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--epochs', default=20, type=int)
  parser.add_argument('--num_units', default=256, type=int)
  parser.add_argument('--dropout', default=0.2, type=float)
  parser.add_argument('--learning_rate', default=0.0001, type=float)
  parser.add_argument('--noise_dim', default=128, type=int)
  parser.add_argument('--summary_freq', default=200, type=int)
  parser.add_argument('--gradient_penalty', default=10.0, type=float)
  parser.add_argument('--generator', default='conv1d', type=str)
  parser.add_argument('--discriminator', default='conv1d', type=str)
  parser.add_argument('--activation', default='tanh', type=str)
  parser.add_argument(
      '--algorithm',
      default='gan',
      type=str,
      help='which alogrithm to train models')
  parser.add_argument(
      '--n_critic',
      default=5,
      type=int,
      help='number of steps between each generator update')
  parser.add_argument(
      '--clear_output_dir',
      action='store_true',
      help='delete output directory if exists')
  parser.add_argument(
      '--keep_generated',
      action='store_true',
      help='keep generated calcium signals and spike trains')
  parser.add_argument(
      '--num_processors',
      default=6,
      type=int,
      help='number of processing cores to use for metrics calculation')
  parser.add_argument(
      '--skip_spike_metrics',
      action='store_true',
      help='flag to skip calculating spike metrics')
  parser.add_argument(
      '--plot_weights',
      action='store_true',
      help='flag to plot weights and activations in TensorBoard')
  parser.add_argument(
      '--skip_checkpoint',
      action='store_true',
      help='flag to skip storing model checkpoints')
  parser.add_argument('--verbose', default=1, type=int)
  hparams = parser.parse_args()

  hparams.global_step = 0

  main(hparams)
