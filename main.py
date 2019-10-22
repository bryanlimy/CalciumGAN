import argparse
import numpy as np
from time import time
from tqdm import tqdm
import tensorflow as tf

np.random.seed(1234)
tf.random.set_seed(1234)

from utils import get_dataset, Summary
from models import get_generator, get_discriminator


def gradient_penalty(inputs, generated, discriminator):
  shape = (inputs.shape[0],) + ((1,) * (len(inputs.shape) - 1))
  epsilon = tf.random.uniform(shape, minval=0.0, maxval=1.0)
  x_hat = epsilon * inputs + (1 - epsilon) * generated
  with tf.GradientTape() as tape:
    tape.watch(x_hat)
    d_hat = discriminator(x_hat, training=True)
  gradients = tape.gradient(d_hat, x_hat)
  slopes = tf.sqrt(
      tf.reduce_mean(
          tf.square(gradients), axis=list(range(len(inputs.shape)))[1:]))
  penalty = tf.reduce_mean(tf.square(slopes - 1.0))
  return penalty


def compute_loss(inputs,
                 generator,
                 discriminator,
                 noise_dim,
                 penalty_weight=10.0,
                 training=True):
  noise = tf.random.normal((inputs.shape[0], noise_dim))

  generated = generator(noise, training=training)

  real = discriminator(inputs, training=training)
  fake = discriminator(generated, training=training)

  penalty = gradient_penalty(inputs, generated, discriminator)
  dis_loss = tf.reduce_mean(real) - tf.reduce_mean(
      fake) + penalty_weight * penalty
  gen_loss = tf.reduce_mean(fake)
  return gen_loss, dis_loss, penalty


@tf.function
def train_step(inputs, generator, discriminator, gen_optimizer, dis_optimizer,
               noise_dim, penalty_weight):

  with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
    gen_loss, dis_loss, penalty = compute_loss(
        inputs,
        generator,
        discriminator,
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

  for inputs in tqdm(
      train_ds,
      desc='Epoch {:02d}/{:02d}'.format(epoch + 1, hparams.epochs),
      total=hparams.steps_per_epoch):

    gen_loss, dis_loss, penalty = train_step(
        inputs,
        generator,
        discriminator,
        gen_optimizer,
        dis_optimizer,
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
def validation_step(inputs, generator, discriminator, noise_dim,
                    penalty_weight):

  gen_loss, dis_loss, penalty = compute_loss(
      inputs,
      generator,
      discriminator,
      noise_dim=noise_dim,
      penalty_weight=penalty_weight,
      training=False)

  return gen_loss, dis_loss, penalty


def validate(hparams, validation_ds, generator, discriminator, summary):
  gen_losses, dis_losses, penalties = [], [], []

  for inputs in validation_ds:
    gen_loss, dis_loss, penalty = validation_step(
        inputs,
        generator,
        discriminator,
        noise_dim=hparams.noise_dim,
        penalty_weight=hparams.gradient_penalty)

    gen_losses.append(gen_loss)
    dis_losses.append(dis_loss)
    penalties.append(penalty)

  gen_losses, dis_losses = np.mean(gen_losses), np.mean(dis_losses)

  summary.scalar('generator_loss', gen_losses, training=False)
  summary.scalar('discriminator_loss', dis_losses, training=False)
  summary.scalar('gradient_penalty', np.mean(penalties), training=False)

  return gen_losses, dis_losses


def train_and_validate(hparams, train_ds, validation_ds, generator,
                       discriminator, gen_optimizer, dis_optimizer, summary):

  # noise to test generator and plot to TensorBoard
  test_noise = tf.random.normal((5, hparams.noise_dim))

  for epoch in range(hparams.epochs):

    train_gen_loss, train_dis_loss, elapse = train(
        hparams, train_ds, generator, discriminator, gen_optimizer,
        dis_optimizer, summary, epoch)

    val_gen_loss, val_dis_loss = validate(hparams, validation_ds, generator,
                                          discriminator, summary)

    test_generation = generator(test_noise, training=False)
    if hparams.input == 'fashion_mnist':
      summary.image('fake', test_generation, training=False)
    else:
      summary.plot('fake', test_generation, training=False)

    print('Train generator loss {:.4f} Train discriminator loss {:.4f} '
          'Time {:.2f}s\nEval generator loss {:.4f} '
          'Eval discriminator loss {:.4f}\n'.format(train_gen_loss,
                                                    train_dis_loss, elapse,
                                                    val_gen_loss, val_dis_loss))

    summary.scalar('elapse (s)', elapse, step=epoch, training=True)


def main(hparams):
  hparams.global_step = 0

  summary = Summary(hparams)

  train_ds, validation_ds = get_dataset(hparams, summary)

  gen_optimizer = tf.keras.optimizers.Adam(hparams.lr)
  dis_optimizer = tf.keras.optimizers.Adam(hparams.lr)

  generator = get_generator(hparams)
  discriminator = get_discriminator(hparams)

  generator.summary()
  discriminator.summary()

  train_and_validate(hparams, train_ds, validation_ds, generator, discriminator,
                     gen_optimizer, dis_optimizer, summary)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', default='dataset/dataset.pkl')
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
  hparams = parser.parse_args()
  main(hparams)
