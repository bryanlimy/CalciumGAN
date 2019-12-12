import os
import argparse
import numpy as np
from time import time
from tqdm import tqdm
import tensorflow as tf
from shutil import rmtree

np.random.seed(1234)
tf.random.set_seed(1234)

from models.registry import get_model
from utils.summary_helper import Summary
from utils.dataset_helper import get_dataset
from utils.utils import store_hparams, save_signals, get_spike_metrics


def gradient_penalty(inputs, generated, discriminator, training=True):
  shape = (inputs.shape[0],) + (1,) * (len(inputs.shape) - 1)
  epsilon = tf.random.uniform(shape, minval=0.0, maxval=1.0)
  x_hat = epsilon * inputs + (1 - epsilon) * generated
  with tf.GradientTape() as tape:
    tape.watch(x_hat)
    d_hat = discriminator(x_hat, training=training)
  gradients = tape.gradient(d_hat, x_hat)
  axis = list(range(len(inputs.shape)))[1:]
  slope = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=axis))
  penalty = tf.reduce_mean(tf.square(slope - 1.0))
  return penalty


def compute_loss(inputs,
                 generator,
                 discriminator,
                 num_neurons,
                 noise_dim,
                 penalty_weight=10.0,
                 training=True):
  noise = tf.random.normal((inputs.shape[0], num_neurons, noise_dim))

  generated = generator(noise, training=training)

  real = discriminator(inputs, training=training)
  fake = discriminator(generated, training=training)

  penalty = gradient_penalty(
      inputs, generated, discriminator, training=training)
  dis_loss = tf.reduce_mean(real) - tf.reduce_mean(
      fake) + penalty_weight * penalty
  gen_loss = tf.reduce_mean(fake)

  kl_divergence = tf.reduce_mean(
      tf.keras.losses.KLD(y_true=inputs, y_pred=generated))

  return generated, gen_loss, dis_loss, penalty, kl_divergence


@tf.function
def train_step(inputs,
               generator,
               discriminator,
               gen_optimizer,
               dis_optimizer,
               num_neurons,
               noise_dim=64,
               penalty_weight=10.0):

  with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
    _, gen_loss, dis_loss, penalty, _ = compute_loss(
        inputs,
        generator=generator,
        discriminator=discriminator,
        num_neurons=num_neurons,
        noise_dim=noise_dim,
        penalty_weight=penalty_weight,
        training=True)

  gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
  dis_gradients = dis_tape.gradient(dis_loss, discriminator.trainable_variables)

  gen_optimizer.apply_gradients(
      zip(gen_gradients, generator.trainable_variables))
  dis_optimizer.apply_gradients(
      zip(dis_gradients, discriminator.trainable_variables))

  return gen_loss, dis_loss, penalty


def train(hparams, train_ds, generator, discriminator, gen_optimizer,
          dis_optimizer, summary, epoch):
  gen_losses, dis_losses = [], []

  start = time()

  for signal, spike in tqdm(
      train_ds,
      desc='Epoch {:02d}/{:02d}'.format(epoch + 1, hparams.epochs),
      total=hparams.steps_per_epoch):

    gen_loss, dis_loss, penalty = train_step(
        signal,
        generator=generator,
        discriminator=discriminator,
        gen_optimizer=gen_optimizer,
        dis_optimizer=dis_optimizer,
        num_neurons=hparams.num_neurons,
        noise_dim=hparams.noise_dim,
        penalty_weight=hparams.gradient_penalty)

    if hparams.global_step % hparams.summary_freq == 0:
      summary.scalar('generator_loss', gen_loss, training=True)
      summary.scalar('discriminator_loss', dis_loss, training=True)
      summary.scalar('gradient_penalty', penalty, training=True)

    gen_losses.append(gen_loss)
    dis_losses.append(dis_loss)

    hparams.global_step += 1

  end = time()

  return np.mean(gen_losses), np.mean(dis_losses), end - start


@tf.function
def validation_step(inputs, generator, discriminator, noise_dim, num_neurons,
                    penalty_weight):

  generated, gen_loss, dis_loss, penalty, kl_divergence = compute_loss(
      inputs,
      generator,
      discriminator,
      num_neurons=num_neurons,
      noise_dim=noise_dim,
      penalty_weight=penalty_weight,
      training=False)

  return generated, gen_loss, dis_loss, penalty, kl_divergence


def validate(hparams, validation_ds, generator, discriminator, summary, epoch):
  gen_losses, dis_losses, penalties, kl_divergences = [], [], [], []

  start = time()

  i = 0
  for signal, spike in validation_ds:
    generated, gen_loss, dis_loss, penalty, kl_divergence = validation_step(
        signal,
        generator,
        discriminator,
        num_neurons=hparams.num_neurons,
        noise_dim=hparams.noise_dim,
        penalty_weight=hparams.gradient_penalty)

    gen_losses.append(gen_loss)
    dis_losses.append(dis_loss)
    penalties.append(penalty)
    kl_divergences.append(kl_divergence)

    save_signals(hparams, epoch, signal.numpy(), spike.numpy(),
                 generated.numpy())

  gen_losses, dis_losses = np.mean(gen_losses), np.mean(dis_losses)

  mean_spike_error, mean_van_rossum_distance = get_spike_metrics(hparams, epoch)

  end = time()

  summary.scalar('generator_loss', gen_losses, training=False)
  summary.scalar('discriminator_loss', dis_losses, training=False)
  summary.scalar('gradient_penalty', np.mean(penalties), training=False)
  summary.scalar('kl_divergence', np.mean(kl_divergences), training=False)
  summary.scalar('elapse (s)', end - start, step=epoch, training=False)
  summary.scalar('mean_spike_error', mean_spike_error, training=False)
  summary.scalar('mean_van_rossum', mean_van_rossum_distance, training=False)

  return gen_losses, dis_losses


def train_and_validate(hparams, train_ds, validation_ds, generator,
                       discriminator, gen_optimizer, dis_optimizer, summary):

  # noise to test generator and plot to TensorBoard
  test_noise = tf.random.normal((1, hparams.num_neurons, hparams.noise_dim))

  for epoch in range(hparams.epochs):

    train_gen_loss, train_dis_loss, elapse = train(
        hparams,
        train_ds,
        generator=generator,
        discriminator=discriminator,
        gen_optimizer=gen_optimizer,
        dis_optimizer=dis_optimizer,
        summary=summary,
        epoch=epoch)

    val_gen_loss, val_dis_loss = validate(
        hparams,
        validation_ds,
        generator=generator,
        discriminator=discriminator,
        summary=summary,
        epoch=epoch)

    test_generation = generator(test_noise, training=False)
    if hparams.input == 'fashion_mnist':
      summary.image('fake', signals=test_generation, training=False)
    else:
      summary.plot_traces('fake', signals=test_generation, training=False)

    print('Train generator loss {:.4f} Train discriminator loss {:.4f} '
          'Time {:.2f}s\nEval generator loss {:.4f} '
          'Eval discriminator loss {:.4f}\n'.format(train_gen_loss,
                                                    train_dis_loss, elapse,
                                                    val_gen_loss, val_dis_loss))

    summary.scalar('elapse (s)', elapse, step=epoch, training=True)


def main(hparams):
  if hparams.clear_output_dir and os.path.exists(hparams.output_dir):
    rmtree(hparams.output_dir)

  hparams.global_step = 0

  summary = Summary(hparams)

  train_ds, validation_ds = get_dataset(hparams)

  gen_optimizer = tf.keras.optimizers.Adam(hparams.lr)
  dis_optimizer = tf.keras.optimizers.Adam(hparams.lr)

  generator, discriminator = get_model(hparams)

  generator.summary()
  discriminator.summary()

  store_hparams(hparams)

  train_and_validate(
      hparams,
      train_ds=train_ds,
      validation_ds=validation_ds,
      generator=generator,
      discriminator=discriminator,
      gen_optimizer=gen_optimizer,
      dis_optimizer=dis_optimizer,
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
  hparams = parser.parse_args()
  main(hparams)
