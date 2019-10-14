import argparse
import numpy as np
from time import time
from tqdm import tqdm
import tensorflow as tf

np.random.seed(1234)
tf.random.set_seed(1234)

from utils import get_dataset, Summary
from models import get_generator, get_discriminator


def generator_loss(fake):
  return tf.keras.losses.binary_crossentropy(
      y_true=tf.ones_like(fake), y_pred=fake, from_logits=True)


def discriminator_loss(real, fake):
  real_loss = tf.keras.losses.binary_crossentropy(
      y_true=tf.ones_like(real), y_pred=real, from_logits=True)
  fake_loss = tf.keras.losses.binary_crossentropy(
      y_true=tf.zeros_like(fake), y_pred=fake, from_logits=True)
  return real_loss + fake_loss


@tf.function
def train_step(inputs, noise_dim, generator, discriminator, gen_optimizer,
               dis_optimizer):
  noise = tf.random.normal((inputs.shape[0], noise_dim))

  with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
    generated = generator(noise, training=True)

    real_output = discriminator(inputs, training=True)
    fake_output = discriminator(generated, training=True)

    gen_loss = generator_loss(fake_output)
    dis_loss = discriminator_loss(real_output, fake_output)

  gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
  dis_gradients = dis_tape.gradient(dis_loss, discriminator.trainable_variables)

  gen_optimizer.apply_gradients(
      zip(gen_gradients, generator.trainable_variables))
  dis_optimizer.apply_gradients(
      zip(dis_gradients, discriminator.trainable_variables))

  return gen_loss, dis_loss


def train(hparams, train_ds, generator, discriminator, gen_optimizer,
          dis_optimizer, summary, epoch):
  gen_losses, dis_losses = [], []

  start = time()

  for x in tqdm(
      train_ds,
      desc='Epoch {:02d}/{:02d}'.format(epoch + 1, hparams.epochs),
      total=hparams.steps_per_epoch):

    gen_loss, dis_loss = train_step(x, hparams.noise_dim, generator,
                                    discriminator, gen_optimizer, dis_optimizer)

    if hparams.global_step % hparams.summary_freq == 0:
      summary.scalar('generator_loss', tf.reduce_mean(gen_loss), training=True)
      summary.scalar(
          'discriminator_loss', tf.reduce_mean(dis_loss), training=True)

    gen_losses.extend(gen_loss)
    dis_losses.extend(dis_loss)

    hparams.global_step += 1

  end = time()

  return np.mean(gen_losses), np.mean(dis_losses), end - start


@tf.function
def validation_step(inputs, noise_dim, generator, discriminator):
  noise = tf.random.normal((inputs.shape[0], noise_dim))

  generated = generator(noise, training=False)

  real_output = discriminator(inputs, training=False)
  fake_output = discriminator(generated, training=False)

  gen_loss = generator_loss(fake_output)
  dis_loss = discriminator_loss(real_output, fake_output)

  return gen_loss, dis_loss


def validate(hparams, validation_ds, generator, discriminator, summary):
  gen_losses, dis_losses = [], []

  for x in validation_ds:
    gen_loss, dis_loss = validation_step(x, hparams.noise_dim, generator,
                                         discriminator)
    gen_losses.extend(gen_loss)
    dis_losses.extend(dis_loss)

  gen_losses, dis_losses = np.mean(gen_losses), np.mean(dis_losses)

  summary.scalar('generator_loss', gen_losses, training=False)
  summary.scalar('discriminator_loss', dis_losses, training=False)

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
    summary.plot('activity', test_generation, training=False)

    print('Train generator loss {:.4f} Train discriminator loss {:.4f} '
          'Time {:.2f}s\nEval generator loss {:.4f} '
          'Eval discriminator loss {:.4f}\n'.format(
              train_gen_loss, train_dis_loss, val_gen_loss, val_dis_loss,
              elapse))

    summary.scalar('elapse (s)', elapse, training=True)


def main(hparams):
  train_ds, validation_ds = get_dataset(hparams)

  summary = Summary(hparams)

  gen_optimizer = tf.keras.optimizers.Adam(hparams.lr)
  dis_optimizer = tf.keras.optimizers.Adam(hparams.lr)

  generator = get_generator(hparams)
  discriminator = get_discriminator(hparams)

  generator.summary()
  discriminator.summary()

  hparams.global_step = 0

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
  parser.add_argument('--lr', default=0.001, type=float)
  parser.add_argument('--noise_dim', default=200, type=int)
  parser.add_argument('--summary_freq', default=200, type=int)
  parser.add_argument('--verbose', default=1, type=int)
  hparams = parser.parse_args()
  main(hparams)
