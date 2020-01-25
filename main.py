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
      total=hparams.steps_per_epoch):

    gen_loss, dis_loss, gradient_penalty, kl = gan.train(signal)

    if hparams.global_step % hparams.summary_freq == 0:
      summary.scalar('generator_loss', gen_loss, training=True)
      summary.scalar('discriminator_loss', dis_loss, training=True)
      if gradient_penalty is not None:
        summary.scalar('gradient_penalty', gradient_penalty, training=True)
      summary.scalar('kl_divergence', kl, training=True)
      if hparams.plot_weights:
        summary.plot_weights(gan, training=True)

    gen_losses.append(gen_loss)
    dis_losses.append(dis_loss)

    hparams.global_step += 1

  end = time()

  summary.scalar('elapse (s)', end - start, step=epoch, training=True)

  return np.mean(gen_losses), np.mean(dis_losses)


def validate(hparams, validation_ds, gan, summary, epoch):
  gen_losses, dis_losses, gradient_penalties, kl_divergences = [], [], [], []

  start = time()

  for signal, spike in validation_ds:
    fake, gen_loss, dis_loss, gradient_penalty, kl = gan.validate(signal)

    gen_losses.append(gen_loss)
    dis_losses.append(dis_loss)
    if gradient_penalty is not None:
      gradient_penalties.append(gradient_penalty)
    kl_divergences.append(kl)

    save_signals(
        hparams,
        epoch,
        real_signals=signal.numpy(),
        real_spikes=spike.numpy(),
        fake_signals=fake.numpy())

  end = time()

  gen_losses, dis_losses = np.mean(gen_losses), np.mean(dis_losses)

  # evaluate spike metrics every 5 epochs
  if not hparams.skip_spike_metrics and (epoch % 5 == 0 or
                                         epoch == hparams.epochs - 1):
    measure_spike_metrics(hparams, epoch, summary)

  # delete generated signals and spike train
  if not hparams.keep_generated:
    delete_generated_file(hparams, epoch)

  summary.scalar('generator_loss', gen_losses, training=False)
  summary.scalar('discriminator_loss', dis_losses, training=False)
  if gradient_penalties:
    summary.scalar(
        'gradient_penalty', np.mean(gradient_penalties), training=False)
  summary.scalar('kl_divergence', np.mean(kl_divergences), training=False)
  summary.scalar('elapse (s)', end - start, step=epoch, training=False)

  return gen_losses, dis_losses


def train_and_validate(hparams, train_ds, validation_ds, gan, summary):

  # noise to test generator and plot to TensorBoard
  test_noise = tf.random.normal((1, hparams.num_neurons, hparams.noise_dim))

  for epoch in range(hparams.epochs):

    train_gen_loss, train_dis_loss = train(
        hparams, train_ds, gan=gan, summary=summary, epoch=epoch)

    val_gen_loss, val_dis_loss = validate(
        hparams, validation_ds, gan=gan, summary=summary, epoch=epoch)

    # test generated data and plot in TensorBoard
    fake = gan.samples(test_noise)
    if hparams.input == 'fashion_mnist':
      summary.image('fake', signals=fake, training=False)
    else:
      summary.plot_traces('fake', signals=fake, training=False)

    print('Train: generator loss {:.4f} discriminator loss {:.4f}\n'
          'Eval: generator loss {:.4f} discriminator loss {:.4f}\n'.format(
              train_gen_loss, train_dis_loss, val_gen_loss, val_dis_loss))

    if epoch % 5 == 0 or epoch == hparams.epochs - 1:
      save_models(hparams, gan, epoch)


def main(hparams):
  if hparams.clear_output_dir and os.path.exists(hparams.output_dir):
    rmtree(hparams.output_dir)

  summary = Summary(hparams)

  train_ds, validation_ds = get_dataset(hparams, summary)

  generator, discriminator = get_models(hparams)

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


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', default='dataset/tfrecords')
  parser.add_argument('--output_dir', default='runs')
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--epochs', default=20, type=int)
  parser.add_argument('--num_units', default=256, type=int)
  parser.add_argument('--dropout', default=0.2, type=float)
  parser.add_argument('--learning_rate', default=0.0001, type=float)
  parser.add_argument('--noise_dim', default=200, type=int)
  parser.add_argument('--summary_freq', default=200, type=int)
  parser.add_argument('--gradient_penalty', default=10.0, type=float)
  parser.add_argument('--verbose', default=1, type=int)
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
  hparams = parser.parse_args()
  hparams.global_step = 0
  main(hparams)
