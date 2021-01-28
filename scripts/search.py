import os
import argparse
import warnings
import numpy as np
from time import time
import tensorflow as tf
from shutil import rmtree
from tensorboard.plugins.hparams import api as hp

np.random.seed(1234)
tf.random.set_seed(1234)

from main import main as train


class HParams(object):

  def __init__(self, args, session, algorithm, model, activation, noise_dim,
               num_units, kernel_size, strides, phase_shuffle, gradient_penalty,
               n_critic):
    self.session = session
    self.input_dir = args.input_dir
    self.output_dir = os.path.join(
        args.output_dir, '{:03d}_{}_units{}_kl{}_strides{}_ps{}_{}_nd{}'.format(
            session, model, num_units, kernel_size, strides, phase_shuffle,
            activation, noise_dim))
    self.batch_size = args.batch_size
    self.num_units = num_units
    self.kernel_size = kernel_size
    self.strides = strides
    self.phase_shuffle = phase_shuffle
    self.epochs = args.epochs
    self.dropout = 0.2
    self.learning_rate = 0.0001
    self.noise_dim = noise_dim
    self.gradient_penalty = gradient_penalty
    self.model = model
    self.activation = activation
    self.batch_norm = False
    self.layer_norm = True
    self.algorithm = algorithm
    self.n_critic = n_critic
    self.clear_output_dir = False
    self.save_generated = 'last'
    self.plot_weights = False
    self.skip_checkpoints = False
    self.mixed_precision = args.mixed_precision
    self.profile = False
    self.dpi = 120
    self.verbose = args.verbose
    self.global_step = 0

    self.surrogate_ds = True if 'surrogate' in args.input_dir else False


def print_experiment_settings(session, hparams):
  print('\nExperiment {:03d}'
        '\n-----------------------------------------\n'
        '\talgorithm: {}\n'
        '\tmodel: {}\n'
        '\tnum_units: {}\n'
        '\tkernel_size: {}\n'
        '\tstrides: {}\n'
        '\tphase shuffle: {}\n'
        '\tactivation: {}\n'
        '\tnoise dim: {}'.format(session, hparams.algorithm, hparams.model,
                                 hparams.num_units, hparams.kernel_size,
                                 hparams.strides, hparams.phase_shuffle,
                                 hparams.activation, hparams.noise_dim))


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

  hp_algorithm = hp.HParam('algorithm', hp.Discrete(['wgan-gp']))
  hp_model = hp.HParam('models', hp.Discrete(['wavegan']))
  hp_activation = hp.HParam('activation', hp.Discrete(['leakyrelu']))
  hp_noise_dim = hp.HParam('noise_dim', hp.Discrete([4, 8, 16]))
  hp_num_units = hp.HParam('num_units', hp.Discrete([8, 16, 32]))
  hp_kernel_size = hp.HParam('kernel_size', hp.Discrete([2, 3, 4]))
  hp_strides = hp.HParam('strides', hp.Discrete([1]))
  hp_phase_shuffle = hp.HParam('phase_shuffle', hp.Discrete([0, 1]))
  hp_gradient_penalty = hp.HParam('gradient_penalty', hp.Discrete([10.0]))
  hp_n_critic = hp.HParam('n_critic', hp.Discrete([5]))

  with tf.summary.create_file_writer(args.output_dir).as_default():
    hp.hparams_config(
        hparams=[
            hp_algorithm, hp_model, hp_activation, hp_noise_dim, hp_num_units,
            hp_kernel_size, hp_strides, hp_phase_shuffle, hp_gradient_penalty,
            hp_n_critic
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
            for num_units in hp_num_units.domain.values:
              for kernel_size in hp_kernel_size.domain.values:
                for strides in hp_strides.domain.values:
                  for phase_shuffle in hp_phase_shuffle.domain.values:
                    for gradient_penalty in hp_gradient_penalty.domain.values:
                      for n_critic in hp_n_critic.domain.values:
                        session += 1

                        hparams = HParams(
                            args,
                            session,
                            algorithm=algorithm,
                            model=model,
                            activation=activation,
                            noise_dim=noise_dim,
                            num_units=num_units,
                            kernel_size=kernel_size,
                            strides=strides,
                            phase_shuffle=phase_shuffle,
                            gradient_penalty=gradient_penalty,
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
                            hp_num_units: num_units,
                            hp_kernel_size: kernel_size,
                            hp_strides: strides,
                            hp_phase_shuffle: phase_shuffle,
                            hp_gradient_penalty: gradient_penalty,
                            hp_n_critic: n_critic
                        }

                        try:
                          start = time()
                          run_experiment(hparams, hp_hparams)
                          end = time()
                          print('\nExperiment {:03d} completed in {:.2f}hrs\n'.
                                format(session, (end - start) / (60 * 60)))
                        except Exception as e:
                          print('\nExperiment {:03d} ERROR: {}'.format(
                              session, e))

  print('\nExperiment completed, TensorBoard log at {}'.format(args.output_dir))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', default='dataset/')
  parser.add_argument('--output_dir', default='runs/hparams_turning')
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--epochs', default=400, type=int)
  parser.add_argument('--clear_output_dir', action='store_true')
  parser.add_argument('--mixed_precision', action='store_true')
  parser.add_argument('--verbose', default=0, type=int)
  args = parser.parse_args()

  if args.verbose == 0:
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

  search(args)
