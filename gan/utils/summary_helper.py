import os
import io
import shutil
import platform
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
if platform.system() == 'Darwin':
  matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

import seaborn as sns

from . import utils, spike_helper


class Summary(object):
  """ 
  Log tf.Summary to output_dir during training and output_dir/eval during 
  evaluation
  """

  def __init__(self, hparams, policy=None, spike_metrics=False):
    self._hparams = hparams
    self.spike_metrics = spike_metrics

    if not spike_metrics:
      self._train_dir = hparams.output_dir
      self._validation_dir = os.path.join(
          os.path.join(hparams.output_dir, 'validation'))

      self._profiler_dir = os.path.join(
          os.path.join(hparams.output_dir, 'profiler'))

      self.train_writer = tf.summary.create_file_writer(self._train_dir)
      self.val_writer = tf.summary.create_file_writer(self._validation_dir)

      self._policy = policy
      self._plot_weights = hparams.plot_weights
    else:
      # for spike metrics
      self._metrics_dir = os.path.join(hparams.output_dir, 'metrics')
      self.metrics_writer = tf.summary.create_file_writer(self._metrics_dir)

      # save plots as vector pdf
      self._vector_dir = os.path.join(self._metrics_dir, 'plots')
      if os.path.exists(self._vector_dir):
        shutil.rmtree(self._vector_dir)
      os.makedirs(self._vector_dir)

    self.tick_size = 15
    plt.rc('xtick', labelsize=self.tick_size)
    plt.rc('ytick', labelsize=self.tick_size)

    self.dpi = hparams.dpi
    self.fontsize = 28
    self.real_color = 'dodgerblue'
    self.fake_color = 'orangered'

  def _get_writer(self, training):
    if self.spike_metrics:
      return self.metrics_writer
    else:
      return self.train_writer if training else self.val_writer

  def _get_loss_scale(self):
    return self._policy.loss_scale._current_loss_scale if self._policy else None

  def _plot_to_png(self):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, dpi=self.dpi, format='png')
    buf.seek(0)
    return tf.image.decode_png(buf.getvalue(), channels=4)

  def save_vector_plot(self, filename):
    if self.spike_metrics:
      plt.tight_layout()
      plt.savefig(
          os.path.join(self._vector_dir, '{}.pdf'.format(filename)),
          dpi=self.dpi,
          format='pdf')

  def scalar(self, tag, value, step=0, training=True):
    writer = self._get_writer(training)
    with writer.as_default():
      tf.summary.scalar(tag, value, step=step)

  def histogram(self, tag, values, step=0, training=True):
    writer = self._get_writer(training)
    with writer.as_default():
      tf.summary.histogram(tag, values, step=step)

  def image(self, tag, values, step=0, training=True):
    if type(values) == list:
      values = tf.stack(values)
    writer = self._get_writer(training)
    with writer.as_default():
      tf.summary.image(tag, data=values, step=step, max_outputs=values.shape[0])

  def profiler_trace(self):
    tf.summary.trace_on(graph=True, profiler=True)

  def profiler_export(self):
    tf.summary.trace_export(name='models', profiler_outdir=self._profiler_dir)

  def plot_traces(self, tag, signals, spikes, indexes, step=0, training=True):
    assert len(signals.shape) == 2 and len(spikes.shape) == 2

    images = []

    if tf.is_tensor(signals):
      signals = signals.numpy()
    if tf.is_tensor(spikes):
      spikes = spikes.numpy()

    # calculate the number of rows needed in subplots
    num_rows, rem = divmod(len(indexes), 3)
    if rem > 0:
      num_rows += 1

    fig = plt.figure(figsize=(32, 4 * num_rows))
    fig.patch.set_facecolor('white')

    plt.tick_params(axis='both', which='minor', labelsize=20)

    for i, neuron in enumerate(indexes):
      plt.subplot(num_rows, 3, i + 1)
      # plot signal
      plt.plot(signals[neuron], label='signal', alpha=0.6, color='dodgerblue')
      # plot spike
      x = np.nonzero(spikes[neuron])[0]
      y = np.zeros(x.shape)
      plt.scatter(x, y, s=400, marker='|', label='spike', color='orangered')

      plt.legend(ncol=3, frameon=False, loc=(.04, .85))
      plt.title('Neuron #{:03d}'.format(neuron), fontsize=self.fontsize)
      plt.xlabel('Time (ms)', fontsize=self.fontsize)

      axis = plt.gca()
      axis.spines['top'].set_visible(False)
      axis.spines['right'].set_visible(False)
      axis.get_xaxis().tick_bottom()
      axis.get_yaxis().tick_left()

    plt.tight_layout()
    images.append(self._plot_to_png())
    self.save_vector_plot(tag)
    plt.close()

    self.image(tag, values=images, step=step, training=training)

  def raster_plot(self,
                  tag,
                  real_spikes,
                  fake_spikes,
                  xlabel='',
                  ylabel='',
                  title=None,
                  step=0,
                  training=True):
    images = []

    real_x, real_y = np.nonzero(real_spikes)
    fake_x, fake_y = np.nonzero(fake_spikes)

    df = pd.DataFrame({
        'x': np.concatenate([real_y, fake_y]),
        'y': np.concatenate([real_x, fake_x]),
        'real_data': [True] * len(real_x) + [False] * len(fake_x)
    })

    g = sns.JointGrid(x='x', y='y', data=df)
    plt.gcf().set_size_inches(24, 14)
    plt.gcf().set_facecolor("white")

    real = df.loc[df.real_data == True]
    fake = df.loc[df.real_data == False]

    sns.scatterplot(
        real.x,
        real.y,
        marker='|',
        color=self.real_color,
        alpha=0.9,
        ax=g.ax_joint)
    ax = sns.scatterplot(
        fake.x,
        fake.y,
        marker='|',
        color=self.fake_color,
        alpha=0.9,
        ax=g.ax_joint)
    ax.set_xlabel(xlabel, fontsize=self.fontsize)
    ax.set_ylabel(ylabel, fontsize=self.fontsize)

    hist_kws = {"rwidth": 0.85, "alpha": 0.6}

    bins = range(0, max(max(real.x), max(fake.x)), 50)
    sns.distplot(
        real.x,
        kde=False,
        hist_kws=hist_kws,
        color=self.real_color,
        ax=g.ax_marg_x,
        bins=bins)
    ax = sns.distplot(
        fake.x,
        kde=False,
        hist_kws=hist_kws,
        color=self.fake_color,
        ax=g.ax_marg_x,
        bins=bins)
    ax.set(xlabel='', ylabel='')

    bins = range(0, max(max(real.y), max(fake.y)), 4)
    sns.distplot(
        real.y,
        kde=False,
        hist_kws=hist_kws,
        color=self.real_color,
        ax=g.ax_marg_y,
        bins=bins,
        vertical=True)
    ax = sns.distplot(
        fake.y,
        kde=False,
        hist_kws=hist_kws,
        color=self.fake_color,
        ax=g.ax_marg_y,
        bins=bins,
        vertical=True)
    ax.set(xlabel='', ylabel='')

    g.ax_joint.legend(
        labels=['real', 'fake'],
        ncol=2,
        frameon=False,
        prop={
            'weight': 'regular',
            'size': self.fontsize
        },
        loc='upper right')

    plt.tight_layout()
    images.append(self._plot_to_png())
    self.save_vector_plot(tag)
    plt.close()

    self.image(tag, values=images, step=step, training=training)

  def plot_distribution(self,
                        tag,
                        data,
                        xlabel='',
                        ylabel='',
                        title='',
                        bins=30,
                        step=0,
                        training=False):
    images = []

    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('white')
    ax = sns.distplot(
        data,
        kde=False,
        hist_kws={"rwidth": 0.85},
        color="dodgerblue",
        bins=bins)
    ax.set_xlabel(xlabel, fontsize=self.fontsize)
    ax.set_ylabel(ylabel, fontsize=self.fontsize)

    plt.tight_layout()
    images.append(self._plot_to_png())
    self.save_vector_plot(tag)
    plt.close()

    self.image(tag, values=images, step=step, training=training)

  def plot_histogram(self,
                     tag,
                     data,
                     xlabel='',
                     ylabel='',
                     title=None,
                     step=0,
                     training=False):
    assert type(data) == tuple
    images = []

    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('white')

    hist_kws = {
        "rwidth": 0.85,
        "alpha": 0.6,
        "range":
        [min(min(data[0]), min(data[1])),
         max(max(data[0]), max(data[0]))]
    }

    sns.distplot(
        data[0],
        bins=30,
        kde=False,
        hist_kws=hist_kws,
        color=self.real_color,
        label="Real")
    ax = sns.distplot(
        data[1],
        bins=30,
        kde=False,
        hist_kws=hist_kws,
        color=self.fake_color,
        label="Fake")

    ax.legend(fontsize=15)
    ax.set_xlabel(xlabel, fontsize=self.fontsize)
    ax.set_ylabel(ylabel, fontsize=self.fontsize)

    plt.tight_layout()
    images.append(self._plot_to_png())
    self.save_vector_plot(tag)
    plt.close()

    self.image(tag, values=images, step=step, training=training)

  def plot_histograms_grid(self,
                           tag,
                           data,
                           xlabel='',
                           ylabel='',
                           titles=None,
                           step=0,
                           training=False):
    assert type(data) == list and type(data[0]) == tuple
    images = []

    num_rows, rem = divmod(len(data), 3)
    if rem > 0:
      num_rows += 1

    fig = plt.figure(figsize=(32, 8 * num_rows))
    fig.patch.set_facecolor('white')

    for i in range(len(data)):
      plt.subplot(num_rows, 3, i + 1)

      real, fake = data[i]

      hist_kws = {
          "rwidth": 0.85,
          "alpha": 0.6,
          "range": [min(min(real), min(fake)),
                    max(max(real), max(fake))]
      }

      sns.distplot(
          real,
          bins=30,
          kde=False,
          hist_kws=hist_kws,
          color=self.real_color,
          label="Real")
      ax = sns.distplot(
          fake,
          bins=30,
          kde=False,
          hist_kws=hist_kws,
          color=self.fake_color,
          label="Fake")

      ax.legend(fontsize=15)
      ax.set_xlabel(xlabel, fontsize=self.fontsize)
      ax.set_ylabel(ylabel, fontsize=self.fontsize)
      ax.set_title(titles[i], fontsize=self.fontsize)

    plt.tight_layout()
    images.append(self._plot_to_png())
    self.save_vector_plot(tag)
    plt.close()

    self.image(tag, values=images, step=step, training=training)

  def plot_heatmaps_grid(self,
                         tag,
                         matrix,
                         xlabel='',
                         ylabel='',
                         xticklabels='auto',
                         yticklabels='auto',
                         titles=None,
                         step=0,
                         training=False):
    assert type(matrix) == list and type(matrix[0]) == np.ndarray
    images = []

    num_rows, rem = divmod(len(matrix), 3)
    if rem > 0:
      num_rows += 1

    fig = plt.figure(figsize=(30, 10 * num_rows))
    fig.patch.set_facecolor('white')

    for i in range(len(matrix)):
      plt.subplot(num_rows, 3, i + 1)
      ax = sns.heatmap(
          matrix[i],
          cmap='YlOrRd',
          xticklabels=xticklabels[i] if type(xticklabels) == list else 'auto',
          yticklabels=yticklabels[i] if type(xticklabels) == list else 'auto',
      )
      ax.set_xlabel(xlabel, fontsize=self.fontsize)
      ax.set_ylabel(ylabel, fontsize=self.fontsize)
      ax.set_title(titles[i], fontsize=self.fontsize)

    plt.tight_layout()
    images.append(self._plot_to_png())
    self.save_vector_plot(tag)
    plt.close()

    self.image(tag, values=images, step=step, training=training)

  def variable_summary(self, variable, name=None, step=0, training=True):
    if name is None:
      name = variable.name
    mean = tf.reduce_mean(variable)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
    self.scalar('{}/0_mean'.format(name), mean, step=step, training=training)
    self.scalar(
        '{}/1_stddev'.format(name), stddev, step=step, training=training)
    self.scalar(
        '{}/2_min'.format(name),
        tf.reduce_min(variable),
        step=step,
        training=training)
    self.scalar(
        '{}/3_max'.format(name),
        tf.reduce_max(variable),
        step=step,
        training=training)
    self.histogram(name, variable, step=step, training=training)

  def plot_weights(self, gan, step=0, training=True):
    for i, var in enumerate(gan.generator.trainable_variables):
      self.variable_summary(
          var,
          name='plots_generator/{:02d}/{}'.format(i + 1, var.name),
          step=step,
          training=training,
      )
    for i, var in enumerate(gan.discriminator.trainable_variables):
      self.variable_summary(
          var,
          name='plots_discriminator/{:02d}/{}'.format(i + 1, var.name),
          step=step,
          training=training,
      )

  def log(self,
          gen_loss,
          dis_loss,
          gradient_penalty,
          metrics=None,
          elapse=None,
          gan=None,
          step=0,
          training=True):
    self.scalar('loss/generator', gen_loss, step=step, training=training)
    self.scalar('loss/discriminator', dis_loss, step=step, training=training)
    if gradient_penalty is not None:
      self.scalar(
          'loss/gradient_penalty',
          gradient_penalty,
          step=step,
          training=training)
    if metrics is not None:
      for tag, value in metrics.items():
        self.scalar(tag, value, step=step, training=training)
    if elapse is not None:
      self.scalar('elapse', elapse, step=step, training=training)
    if gan is not None and self._plot_weights:
      self.plot_weights(gan, step=step, training=training)
    if not training and self._policy is not None:
      self.scalar(
          'model/loss_scale',
          self._get_loss_scale(),
          step=step,
          training=training)
