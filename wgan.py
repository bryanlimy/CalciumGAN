import os
import numpy as np
from tqdm import tqdm
from time import time
import tensorflow as tf
from shutil import rmtree

BATCH_SIZE = 128
SHAPES = (28, 28, 1)
OUTPUT_DIR = 'runs/wgan'
NOISE_DIMS = 64
GRADIENT_PENALTY = 10.0
EPOCHS = 100

if os.path.exists(OUTPUT_DIR):
  rmtree(OUTPUT_DIR)

train_summary = tf.summary.create_file_writer(OUTPUT_DIR)
test_summary = tf.summary.create_file_writer(
    os.path.join(OUTPUT_DIR, 'validation'))

(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

STEPS_PER_EPOCHS = int(np.ceil(len(x_train) / BATCH_SIZE))
BUFFER_SIZE = len(x_train)


def preprocess(images):
  shape = images.shape
  images = np.reshape(images, (shape[0], shape[1], shape[2], 1))
  images = images.astype('float32') / 255.0
  return images


x_train = preprocess(x_train)
x_test = preprocess(x_test)

SHAPE = x_train.shape[1:]

train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_ds = tf.data.Dataset.from_tensor_slices(x_test).batch(BATCH_SIZE)


class WGAN(tf.keras.Model):

  def __init__(self, **kwargs):
    super(WGAN, self).__init__()
    self.__dict__.update(kwargs)

    self.gen = tf.keras.Sequential(self.gen)
    self.dis = tf.keras.Sequential(self.dis)

  def generate(self, z):
    return self.gen(z)

  def discriminate(self, x):
    return self.dis(x)

  def gradient_penalty(self, x, x_gen):
    epsilon = tf.random.uniform([x.shape[0], 1, 1, 1], 0.0, 1.0)
    x_hat = epsilon * x + (1 - epsilon) * x_gen
    with tf.GradientTape() as t:
      t.watch(x_hat)
      d_hat = self.discriminate(x_hat)
    gradients = t.gradient(d_hat, x_hat)
    ddx = tf.sqrt(tf.reduce_sum(gradients**2, axis=[1, 2]))
    d_regularizer = tf.reduce_mean((ddx - 1.0)**2)
    return d_regularizer

  def compute_loss(self, x):
    # generating noise from a uniform distribution
    z_samp = tf.random.normal([x.shape[0], 1, 1, self.n_Z])

    # run noise through generator
    x_gen = self.generate(z_samp)
    # discriminate x and x_gen
    logits_x = self.discriminate(x)
    logits_x_gen = self.discriminate(x_gen)

    # gradient penalty
    d_regularizer = self.gradient_penalty(x, x_gen)

    dis_loss = tf.reduce_mean(logits_x) - tf.reduce_mean(
        logits_x_gen) + d_regularizer * self.gradient_penalty_weight

    # losses of fake with label "1"
    gen_loss = tf.reduce_mean(logits_x_gen)

    return dis_loss, gen_loss, d_regularizer

  #@tf.function
  def train(self, x):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
      dis_loss, gen_loss, penalty = self.compute_loss(x)

    # compute gradients
    gen_gradients = gen_tape.gradient(gen_loss, self.gen.trainable_variables)
    dis_gradients = dis_tape.gradient(dis_loss, self.dis.trainable_variables)

    self.gen_optimizer.apply_gradients(
        zip(gen_gradients, self.gen.trainable_variables))
    self.dis_optimizer.apply_gradients(
        zip(dis_gradients, self.dis.trainable_variables))

    with train_summary.as_default():
      tf.summary.scalar(
          'generator_loss', gen_loss, step=self.gen_optimizer.iterations)
      tf.summary.scalar(
          'discriminator_loss', dis_loss, step=self.disc_optimizer.iterations)
      tf.summary.scalar(
          'gradient_penalty', penalty, step=self.gen_optimizer.iterations)


generator = [
    tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu"),
    tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
    tf.keras.layers.Conv2DTranspose(
        filters=64,
        kernel_size=3,
        strides=(2, 2),
        padding="SAME",
        activation="relu"),
    tf.keras.layers.Conv2DTranspose(
        filters=32,
        kernel_size=3,
        strides=(2, 2),
        padding="SAME",
        activation="relu"),
    tf.keras.layers.Conv2DTranspose(
        filters=1,
        kernel_size=3,
        strides=(1, 1),
        padding="SAME",
        activation="sigmoid"),
]

discriminator = [
    tf.keras.layers.InputLayer(input_shape=SHAPE),
    tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=(2, 2), activation="relu"),
    tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, strides=(2, 2), activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1, activation="sigmoid"),
]

gen_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5)
dis_optimizer = tf.keras.optimizers.RMSprop(0.0005)

model = WGAN(
    gen=generator,
    dis=discriminator,
    gen_optimizer=gen_optimizer,
    dis_optimizer=dis_optimizer,
    n_Z=NOISE_DIMS,
    gradient_penalty_weight=GRADIENT_PENALTY)

noises = tf.random.normal((5, 1, 1, NOISE_DIMS))

for epoch in range(EPOCHS):

  start = time()
  for x in tqdm(train_ds, total=STEPS_PER_EPOCHS):
    model.train(x)
  elapse = time() - start

  dis_losses, gen_losses, penalties = [], [], []
  for x in test_ds:
    dis_loss, gen_loss, penalty = model.compute_loss(x)
    dis_losses.append(dis_loss)
    gen_losses.append(gen_loss)
    penalties.append(penalty)

  with train_summary.as_default():
    tf.summary.scalar('elapse (s)', elapse, step=epoch)

  with test_summary.as_default():
    tf.summary.scalar(
        'discriminator_loss',
        np.mean(dis_losses),
        step=disc_optimizer.iterations)
    tf.summary.scalar(
        'generator_loss', np.mean(gen_losses), step=gen_optimizer.iterations)
    tf.summary.scalar(
        'gradient_penalty', np.mean(penalties), step=gen_optimizer.iterations)

  # generate and log sample
  generated = model.gen(noises)
  with test_summary.as_default():
    tf.summary.image(
        'fake',
        generated,
        step=disc_optimizer.iterations,
        max_outputs=generated.shape[0])

  print('Epoch {:02d}/{:02d} Val discriminator loss {:.4f} '
        'Val generator loss {:.4f} Elapse {:.2f}s\n'.format(
            epoch + 1, EPOCHS, np.mean(dis_losses), np.mean(gen_losses),
            elapse))
