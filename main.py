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
from losses.registry import get_losses
from utils.dataset_helper import get_dataset
from utils.utils import store_hparams, save_signals, measure_spike_metrics, \
  save_models, load_models, delete_generated_file


def step(inputs,
         generator,
         discriminator,
         generator_loss,
         discriminator_loss,
         num_neurons,
         noise_dim,
         training=True):
  noise = tf.random.normal((inputs.shape[0], num_neurons, noise_dim))

  generated = generator(noise, training=training)

  real_output = discriminator(inputs, training=training)
  fake_output = discriminator(generated, training=training)

  gen_loss = generator_loss(real_output, fake_output)
  dis_loss, penalty = discriminator_loss(real_output, fake_output,
                                         discriminator, inputs, generated)

  kl_divergence = tf.reduce_mean(
      tf.keras.losses.KLD(y_true=inputs, y_pred=generated))

  return generated, gen_loss, dis_loss, penalty, kl_divergence


@tf.function
def train_step(inputs,
               generator,
               discriminator,
               gen_optimizer,
               dis_optimizer,
               generator_loss,
               discriminator_loss,
               num_neurons,
               noise_dim=64):

  with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
    _, gen_loss, dis_loss, penalty, _ = step(
        inputs,
        generator=generator,
        discriminator=discriminator,
        generator_loss=generator_loss,
        discriminator_loss=discriminator_loss,
        num_neurons=num_neurons,
        noise_dim=noise_dim,
        training=True)

  gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
  dis_gradients = dis_tape.gradient(dis_loss, discriminator.trainable_variables)

  gen_optimizer.apply_gradients(
      zip(gen_gradients, generator.trainable_variables))
  dis_optimizer.apply_gradients(
      zip(dis_gradients, discriminator.trainable_variables))

  return gen_loss, dis_loss, penalty


def train(hparams, train_ds, generator, discriminator, gen_optimizer,
          dis_optimizer, generator_loss, discriminator_loss, summary, epoch):
  gen_losses, dis_losses = [], []

  start = time()

  for signal, spike in tqdm(
      train_ds,
      desc='Epoch {:03d}/{:03d}'.format(epoch, hparams.epochs),
      total=hparams.steps_per_epoch):

    gen_loss, dis_loss, penalty = train_step(
        signal,
        generator=generator,
        discriminator=discriminator,
        gen_optimizer=gen_optimizer,
        dis_optimizer=dis_optimizer,
        generator_loss=generator_loss,
        discriminator_loss=discriminator_loss,
        num_neurons=hparams.num_neurons,
        noise_dim=hparams.noise_dim)

    if hparams.global_step % hparams.summary_freq == 0:
      summary.scalar('generator_loss', gen_loss, training=True)
      summary.scalar('discriminator_loss', dis_loss, training=True)
      summary.scalar('gradient_penalty', penalty, training=True)
      if hparams.plot_weights:
        summary.plot_weights(generator, discriminator, training=True)

    gen_losses.append(gen_loss)
    dis_losses.append(dis_loss)

    hparams.global_step += 1

  end = time()

  return np.mean(gen_losses), np.mean(dis_losses), end - start


@tf.function
def validation_step(inputs, generator, discriminator, generator_loss,
                    discriminator_loss, noise_dim, num_neurons):

  generated, gen_loss, dis_loss, penalty, kl_divergence = step(
      inputs,
      generator,
      discriminator,
      generator_loss=generator_loss,
      discriminator_loss=discriminator_loss,
      num_neurons=num_neurons,
      noise_dim=noise_dim,
      training=False)

  return generated, gen_loss, dis_loss, penalty, kl_divergence


def validate(hparams, validation_ds, generator, discriminator, generator_loss,
             discriminator_loss, summary, epoch):
  gen_losses, dis_losses, penalties, kl_divergences = [], [], [], []

  start = time()

  for signal, spike in validation_ds:
    generated, gen_loss, dis_loss, penalty, kl_divergence = validation_step(
        signal,
        generator,
        discriminator,
        generator_loss=generator_loss,
        discriminator_loss=discriminator_loss,
        num_neurons=hparams.num_neurons,
        noise_dim=hparams.noise_dim)

    gen_losses.append(gen_loss)
    dis_losses.append(dis_loss)
    penalties.append(penalty)
    kl_divergences.append(kl_divergence)

    save_signals(
        hparams,
        epoch,
        real_signals=signal.numpy(),
        real_spikes=spike.numpy(),
        fake_signals=generated.numpy())

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
  summary.scalar('gradient_penalty', np.mean(penalties), training=False)
  summary.scalar('kl_divergence', np.mean(kl_divergences), training=False)
  summary.scalar('elapse (s)', end - start, step=epoch, training=False)

  return gen_losses, dis_losses


def train_and_validate(hparams, train_ds, validation_ds, generator,
                       discriminator, summary):

  # noise to test generator and plot to TensorBoard
  test_noise = tf.random.normal((1, hparams.num_neurons, hparams.noise_dim))

  generator_loss, discriminator_loss = get_losses(hparams)

  gen_optimizer = tf.keras.optimizers.Adam(hparams.lr)
  dis_optimizer = tf.keras.optimizers.Adam(hparams.lr)

  for epoch in range(hparams.epochs):

    train_gen_loss, train_dis_loss, elapse = train(
        hparams,
        train_ds,
        generator=generator,
        discriminator=discriminator,
        generator_loss=generator_loss,
        discriminator_loss=discriminator_loss,
        gen_optimizer=gen_optimizer,
        dis_optimizer=dis_optimizer,
        summary=summary,
        epoch=epoch)

    val_gen_loss, val_dis_loss = validate(
        hparams,
        validation_ds,
        generator=generator,
        discriminator=discriminator,
        generator_loss=generator_loss,
        discriminator_loss=discriminator_loss,
        summary=summary,
        epoch=epoch)

    # test generated data and plot in TensorBoard
    generated = generator(test_noise, training=False)
    if hparams.input == 'fashion_mnist':
      summary.image('fake', signals=generated, training=False)
    else:
      summary.plot_traces('fake', signals=generated, training=False)

    print(
        'Train: generator loss {:.4f} discriminator loss {:.4f} Time {:.2f}s\n'
        'Eval: generator loss {:.4f} discriminator loss {:.4f}\n'.format(
            train_gen_loss, train_dis_loss, elapse, val_gen_loss, val_dis_loss))

    summary.scalar('elapse (s)', elapse, step=epoch, training=True)

    if epoch % 5 == 0 or epoch == hparams.epochs - 1:
      save_models(hparams, generator, discriminator, epoch)


def main(hparams):
  if hparams.clear_output_dir and os.path.exists(hparams.output_dir):
    rmtree(hparams.output_dir)

  hparams.global_step = 0

  summary = Summary(hparams)

  train_ds, validation_ds = get_dataset(hparams, summary)

  generator, discriminator = get_models(hparams)

  generator.summary()
  discriminator.summary()

  store_hparams(hparams)

  load_models(hparams, generator, discriminator)

  train_and_validate(
      hparams,
      train_ds=train_ds,
      validation_ds=validation_ds,
      generator=generator,
      discriminator=discriminator,
      summary=summary)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', default='dataset/tfrecords')
  parser.add_argument('--output_dir', default='runs')
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--epochs', default=20, type=int)
  parser.add_argument('--num_units', default=256, type=int)
  parser.add_argument('--dropout', default=0.2, type=float)
  parser.add_argument('--lr', default=0.0001, type=float)
  parser.add_argument('--noise_dim', default=200, type=int)
  parser.add_argument('--summary_freq', default=200, type=int)
  parser.add_argument('--gradient_penalty', default=10.0, type=float)
  parser.add_argument('--verbose', default=1, type=int)
  parser.add_argument('--generator', default='conv1d', type=str)
  parser.add_argument('--discriminator', default='conv1d', type=str)
  parser.add_argument('--activation', default='tanh', type=str)
  parser.add_argument('--loss', default='gan', type=str)
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
  main(hparams)
