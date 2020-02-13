import os
import argparse
import numpy as np
from time import time
import tensorflow as tf
from shutil import rmtree
from tensorboard.plugins.hparams import api as hp

np.random.seed(1234)
tf.random.set_seed(1234)

from main import main as train


class HParams(object):

  def __init__(self, args, session, dropout, noise_dim, gradient_penalty,
               generator, discriminator, activation, algorithm, n_critic):
    self.session = session
    self.input_dir = args.input_dir
    self.output_dir = os.path.join(
        args.output_dir,
        '{:03d}_{}_{}_{}_{}_{}'.format(session, algorithm, generator,
                                       discriminator, activation, noise_dim))
    self.batch_size = args.batch_size
    self.epochs = args.epochs
    self.dropout = dropout
    self.learning_rate = 0.0001
    self.noise_dim = noise_dim
    self.summary_freq = 200
    self.gradient_penalty = gradient_penalty
    self.generator = generator
    self.discriminator = discriminator
    self.activation = activation
    self.algorithm = algorithm
    self.n_critic = n_critic
    self.clear_output_dir = False
    self.keep_generated = args.keep_generated
    self.num_processors = args.num_processors
    self.spike_metrics = args.skip_spike_metrics
    self.spike_metrics_freq = 20
    self.plot_weights = False
    self.save_checkpoints = False
    self.mixed_precision = args.mixed_precision
    self.verbose = args.verbose
    self.global_step = 0


def print_experiment_settings(session, hparams):
  print('\nExperiment {:03d}'
        '\n-----------------------------------------\n'
        '\talgorithm: {}\n'
        '\tgenerator: {}\n'
        '\tdiscriminator: {}\n'
        '\tactivation: {}\n'
        '\tnoise dim: {}'.format(session, hparams.algorithm, hparams.generator,
                                 hparams.discriminator, hparams.activation,
                                 hparams.noise_dim))


def run_experiment(hparams, hp_hparams):
  print_experiment_settings(hparams.session, hparams)

  logdir = os.path.join(hparams.output_dir, 'test')
  with tf.summary.create_file_writer(logdir).as_default():
    hp.hparams(hp_hparams)
    metrics = train(hparams, return_metrics=True)
    for key, item in metrics.items():
      tf.summary.scalar('test/{}'.format(key), item, step=hparams.epochs + 1)


def search(args):
  if args.clear_output_dir and os.path.exists(args.output_dir):
    rmtree(args.output_dir)

  hp_algorithm = hp.HParam('algorithm', hp.Discrete(['gan', 'wgan-gp']))
  hp_model = hp.HParam('models', hp.Discrete(['mlp']))
  hp_activation = hp.HParam('activation',
                            hp.Discrete(['sigmoid', 'tanh', 'relu']))
  hp_noise_dim = hp.HParam('noise_dim', hp.Discrete([16, 32, 64, 128]))
  hp_dropout = hp.HParam('dropout', hp.Discrete([0.2]))
  hp_gradient_penalty = hp.HParam('gradient_penalty', hp.Discrete([10.0]))
  hp_n_critic = hp.HParam('n_critic', hp.Discrete([5]))

  with tf.summary.create_file_writer(args.output_dir).as_default():
    hp.hparams_config(
        hparams=[
            hp_algorithm, hp_model, hp_activation, hp_noise_dim, hp_dropout,
            hp_gradient_penalty, hp_n_critic
        ],
        metrics=[
            hp.Metric('test/signals_metrics/min', display_name='min'),
            hp.Metric('test/signals_metrics/max', display_name='max'),
            hp.Metric('test/signals_metrics/mean', display_name='mean'),
            hp.Metric('test/signals_metrics/std', display_name='std')
        ])

    session = 0

    for algorithm in hp_algorithm.domain.values:
      for model in hp_model.domain.values:
        for activation in hp_activation.domain.values:
          for noise_dim in hp_noise_dim.domain.values:
            for dropout in hp_dropout.domain.values:
              for gradient_penalty in hp_gradient_penalty.domain.values:
                for n_critic in hp_n_critic.domain.values:
                  session += 1

                  hparams = HParams(
                      args,
                      session,
                      dropout=dropout,
                      noise_dim=noise_dim,
                      gradient_penalty=gradient_penalty,
                      generator=model,
                      discriminator=model,
                      activation=activation,
                      algorithm=algorithm,
                      n_critic=n_critic)

                  if os.path.exists(hparams.output_dir):
                    print('Experiment {} already exists'.format(
                        hparams.output_dir))
                    continue

                  hp_hparams = {
                      hp_algorithm: algorithm,
                      hp_model: model,
                      hp_activation: activation,
                      hp_noise_dim: noise_dim,
                      hp_dropout: dropout,
                      hp_gradient_penalty: gradient_penalty,
                      hp_n_critic: n_critic
                  }

                  try:
                    start = time()
                    run_experiment(hparams, hp_hparams)
                    end = time()
                    print('\nExperiment {:03d} completed in {:.2f}hrs\n'.format(
                        session, (end - start) / (60 * 60)))
                  except Exception as e:
                    print('\nExperiment {:03d} ERROR: {}'.format(session, e))

  print('\nExperiment completed, TensorBoard log at {}'.format(args.output_dir))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', default='dataset/')
  parser.add_argument('--output_dir', default='runs/hparams_turning')
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--epochs', default=100, type=int)
  parser.add_argument(
      '--keep_generated',
      action='store_true',
      help='keep generated calcium signals and spike trains')
  parser.add_argument(
      '--clear_output_dir',
      action='store_true',
      help='delete output directory if exists')
  parser.add_argument(
      '--num_processors',
      default=6,
      type=int,
      help='number of processing cores to use for metrics calculation')
  parser.add_argument(
      '--spike_metrics',
      action='store_true',
      help='flag to calculate spike metrics')
  parser.add_argument(
      '--mixed_precision', action='store_true', help='use mixed precision')
  parser.add_argument('--verbose', default=0, type=int)
  args = parser.parse_args()
  search(args)
