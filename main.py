import os
import argparse
import numpy as np
from time import time
from tqdm import tqdm
import tensorflow as tf
from shutil import rmtree

from tensorflow.keras.mixed_precision import experimental as mixed_precision

np.random.seed(1234)
tf.random.set_seed(1234)

from gan.utils import utils
from gan.models.registry import get_models
from gan.utils.summary_helper import Summary
from gan.utils.dataset_helper import get_dataset
from gan.algorithms.registry import get_algorithm


def set_precision_policy(hparams):
  policy = None
  if hparams.mixed_precision:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    if hparams.verbose:
      print('\nCompute dtype: {}\nVariable dtype: {}\n'.format(
          policy.compute_dtype, policy.variable_dtype))
  return policy


def train(hparams, train_ds, gan, summary, epoch):
  gen_losses, dis_losses = [], []
  batch_count = 0

  start = time()

  for signal, _ in tqdm(
      train_ds,
      desc='Train',
      total=hparams.train_steps,
      disable=not bool(hparams.verbose)):

    if hparams.profile and batch_count == 2 and epoch == 1:
      # profile the training session of the 2nd batch in 2nd epoch
      summary.profiler_trace()

    gen_loss, dis_loss, gradient_penalty, metrics = gan.train(signal)

    if hparams.profile and batch_count == 6 and epoch == 1:
      summary.profiler_export()

    # log training progress every 200 steps
    if hparams.global_step % 200 == 0:
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
    batch_count += 1

  end = time()

  summary.scalar('elapse', end - start, training=True)

  return np.mean(gen_losses), np.mean(dis_losses)


def validate(hparams, validation_ds, gan, summary, epoch):
  gen_losses, dis_losses, gradient_penalties, results = [], [], [], {}

  start = time()

  for signal, _ in tqdm(
      validation_ds,
      desc='Validate',
      total=hparams.validation_steps,
      disable=not bool(hparams.verbose)):
    fake, gen_loss, dis_loss, gradient_penalty, metrics = gan.validate(signal)

    # append losses and metrics
    gen_losses.append(gen_loss)
    dis_losses.append(dis_loss)
    if gradient_penalty is not None:
      gradient_penalties.append(gradient_penalty)
    for key, item in metrics.items():
      if key not in results:
        results[key] = []
      results[key].append(item)

    if hparams.save_generated and (epoch % hparams.save_generated_freq == 0 or
                                   epoch == hparams.epochs - 1):
      utils.save_fake_signals(hparams, epoch, signals=fake.numpy())

  gen_loss, dis_loss = np.mean(gen_losses), np.mean(dis_losses)
  gradient_penalty = np.mean(gradient_penalties) if gradient_penalties else None
  results = {key: np.mean(item) for key, item in results.items()}

  end = time()

  summary.log(
      gen_loss,
      dis_loss,
      gradient_penalty,
      metrics=results,
      elapse=end - start,
      training=False)

  return gen_loss, dis_loss


def train_and_validate(hparams, train_ds, validation_ds, gan, summary):
  # noise to test generator and plot to TensorBoard
  test_noise = gan.get_noise(batch_size=1)

  for epoch in range(hparams.epochs):
    if hparams.verbose:
      print('Epoch {:03d}/{:03d}'.format(epoch, hparams.epochs))

    start = time()

    train_gen_loss, train_dis_loss = train(
        hparams, train_ds, gan=gan, summary=summary, epoch=epoch)

    val_gen_loss, val_dis_loss = validate(
        hparams, validation_ds, gan=gan, summary=summary, epoch=epoch)

    if epoch % 10 == 0 or epoch == hparams.epochs - 1:
      # test generated data and plot in TensorBoard
      summary.plot_traces(
          'fake',
          signals=gan.generate(test_noise, denorm=True),
          step=epoch,
          indexes=hparams.focus_neurons,
          training=False)
      if hparams.save_checkpoints:
        utils.save_models(hparams, gan, epoch)

    end = time()

    if hparams.verbose:
      print('Train: generator loss {:.04f} discriminator loss {:.04f}\n'
            'Eval: generator loss {:.04f} discriminator loss {:.04f}\n'
            'Elapse: {:.02f} mins\n'.format(train_gen_loss, train_dis_loss,
                                            val_gen_loss, val_dis_loss,
                                            (end - start) / 60))


def test(validation_ds, gan):
  gen_losses, dis_losses, results = [], [], {}

  for signal, _ in validation_ds:
    _, gen_loss, dis_loss, _, metrics = gan.validate(signal)

    gen_losses.append(gen_loss)
    dis_losses.append(dis_loss)
    for key, item in metrics.items():
      if key not in results:
        results[key] = []
      results[key].append(item)

  return {key: np.mean(item) for key, item in results.items()}


def main(hparams, return_metrics=False):
  if hparams.clear_output_dir and os.path.exists(hparams.output_dir):
    rmtree(hparams.output_dir)

  tf.keras.backend.clear_session()

  policy = set_precision_policy(hparams)

  summary = Summary(hparams, policy=policy)

  train_ds, validation_ds = get_dataset(hparams, summary)

  generator, discriminator = get_models(hparams, summary)

  if hparams.verbose:
    generator.summary()
    print('')
    discriminator.summary()

  utils.store_hparams(hparams)

  utils.load_models(hparams, generator, discriminator)

  gan = get_algorithm(hparams, generator, discriminator, summary)

  start = time()

  train_and_validate(
      hparams,
      train_ds=train_ds,
      validation_ds=validation_ds,
      gan=gan,
      summary=summary)

  end = time()

  summary.scalar('elapse/total', end - start)

  if return_metrics:
    return test(validation_ds, gan)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', default='dataset/tfrecords')
  parser.add_argument('--output_dir', default='runs')
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--epochs', default=20, type=int)
  parser.add_argument('--dropout', default=0.2, type=float)
  parser.add_argument('--learning_rate', default=0.0001, type=float)
  parser.add_argument('--noise_dim', default=128, type=int)
  parser.add_argument('--gradient_penalty', default=10.0, type=float)
  parser.add_argument('--model', default='mlp', type=str)
  parser.add_argument('--activation', default='linear', type=str)
  parser.add_argument('--batch_norm', action='store_true')
  parser.add_argument('--algorithm', default='gan', type=str)
  parser.add_argument(
      '--n_critic',
      default=5,
      type=int,
      help='number of steps between each generator update')
  parser.add_argument('--clear_output_dir', action='store_true')
  parser.add_argument('--save_generated', action='store_true')
  parser.add_argument('--save_generated_freq', default=10, type=int)
  parser.add_argument('--plot_weights', action='store_true')
  parser.add_argument('--save_checkpoints', action='store_true')
  parser.add_argument('--mixed_precision', action='store_true')
  parser.add_argument(
      '--profile', action='store_true', help='enable TensorBoard profiling')
  parser.add_argument('--focus_neurons', action='store_true')
  parser.add_argument('--verbose', default=1, type=int)
  hparams = parser.parse_args()

  # hand picked neurons to plots
  if hparams.focus_neurons:
    hparams.focus_neurons = [87, 58, 90, 39, 7, 60, 14, 5, 13]

  hparams.global_step = 0

  main(hparams)
