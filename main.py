import argparse
import numpy as np
from time import time
from tqdm import tqdm
import tensorflow as tf

np.random.seed(1234)
tf.random.set_seed(1234)

from utils import get_dataset
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


#@tf.function
def train_step(hparams, inputs, generator, discriminator, gen_optimizer,
               dis_optimizer):
  noise = tf.random.normal((inputs.shape[0], hparams.noise_dim))

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


#@tf.function
def validation_step(hparams, inputs, generator, discriminator):
  noise = tf.random.normal((inputs.shape[0], hparams.noise_dim))

  generated = generator(noise, training=False)

  real_output = discriminator(inputs, training=False)
  fake_output = discriminator(generated, training=False)

  gen_loss = generator_loss(fake_output)
  dis_loss = discriminator_loss(real_output, fake_output)

  return gen_loss, dis_loss


def train(hparams, train_ds, generator, discriminator, gen_optimizer,
          dis_optimizer, epoch):
  gen_losses, dis_losses = [], []

  start = time()

  for x in tqdm(
      train_ds,
      desc='Epoch {:02d}/{:02d}'.format(epoch + 1, hparams.epochs),
      total=hparams.steps_per_epoch):
    gen_loss, dis_loss = train_step(hparams, x, generator, discriminator,
                                    gen_optimizer, dis_optimizer)
    gen_losses.extend(gen_loss)
    dis_losses.extend(dis_loss)

  end = time()

  return np.mean(gen_losses), np.mean(dis_losses), end - start


def validate(hparams, validation_ds, generator, discriminator):
  gen_losses, dis_losses = [], []

  for x in validation_ds:
    gen_loss, dis_loss = validation_step(hparams, x, generator, discriminator)
    gen_losses.extend(gen_loss)
    dis_losses.extend(dis_loss)

  return np.mean(gen_losses), np.mean(dis_losses)


def train_and_validate(hparams, train_ds, validation_ds, generator,
                       discriminator, gen_optimizer, dis_optimizer):
  for epoch in range(hparams.epochs):

    train_gen_loss, train_dis_loss, elapse = train(hparams, train_ds, generator,
                                                   discriminator, gen_optimizer,
                                                   dis_optimizer, epoch)

    val_gen_loss, val_dis_loss = validate(hparams, validation_ds, generator,
                                          discriminator)

    print('Train generator loss {:.4f} Train discriminator loss {:.4f} '
          'Time {:.2f}\nEval generator loss {:.4f} '
          'Eval discriminator loss {:.4f}\n'.format(
              train_gen_loss, train_dis_loss, val_gen_loss, val_dis_loss,
              elapse))


def main(hparams):
  train_ds, validation_ds = get_dataset(hparams)

  gen_optimizer = tf.keras.optimizers.Adam(hparams.lr)
  dis_optimizer = tf.keras.optimizers.Adam(hparams.lr)

  generator = get_generator(hparams)
  discriminator = get_discriminator(hparams)

  generator.summary()
  discriminator.summary()

  train_and_validate(hparams, train_ds, validation_ds, generator, discriminator,
                     gen_optimizer, dis_optimizer)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', default='dataset/dataset.pkl')
  parser.add_argument('--output_dir', default='runs')
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--epochs', default=20, type=int)
  parser.add_argument('--num_units', default=256, type=int)
  parser.add_argument('--dropout', default=0.2, type=float)
  parser.add_argument('--lr', default=0.001, type=float)
  parser.add_argument('--model', default='mlp')
  parser.add_argument('--activation', default='relu')
  parser.add_argument('--noise_dim', default=200, type=int)
  parser.add_argument('--verbose', default=1, type=int)
  hparams = parser.parse_args()
  main(hparams)
