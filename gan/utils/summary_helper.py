import os
import io
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

  def __init__(self, hparams, policy=None):
    self._hparams = hparams
    self._train_dir = hparams.output_dir
    self._validation_dir = os.path.join(
        os.path.join(hparams.output_dir, 'validation'))
    self._profiler_dir = os.path.join(
        os.path.join(hparams.output_dir, 'profiler'))

    self.train_writer = tf.summary.create_file_writer(self._train_dir)
    self.val_writer = tf.summary.create_file_writer(self._validation_dir)

    self._policy = policy
    self._plot_weights = hparams.plot_weights

    self._dpi = hparams.dpi

    # color for matplotlib
    self.real_color = 'dodgerblue'
    self.fake_color = 'orangered'

  def _get_writer(self, training):
    return self.train_writer if training else self.val_writer

  def _get_loss_scale(self):
    return self._policy.loss_scale._current_loss_scale if self._policy else None

  def _plot_to_image(self):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, dpi=self._dpi, format='png')
    buf.seek(0)
    return tf.image.decode_png(buf.getvalue(), channels=4)

  @staticmethod
  def _simple_axis(axis):
    """plot only x and y axis, not a frame for subplot ax"""
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.get_xaxis().tick_bottom()
    axis.get_yaxis().tick_left()

  def _plot_trace(self, signal, spike, neuron=None):
    assert len(signal) == len(spike)

    plt.figure(figsize=(20, 4))
    plt.plot(signal, label='signal', alpha=0.6, color='dodgerblue')

    x = np.nonzero(spike)[0]
    y = np.zeros(x.shape)
    plt.scatter(x, y, s=200, marker='|', label='spike', color='orangered')
    plt.legend(ncol=3, frameon=False, loc=(.04, .85))
    plt.xlabel('Time (ms)')
    if neuron is not None:
      plt.title('Neuron #{:03d}'.format(neuron))
    axis = plt.gca()
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.get_xaxis().tick_bottom()
    axis.get_yaxis().tick_left()
    plt.tight_layout()
    image = self._plot_to_image()
    plt.close()
    return image

  def scalar(self, tag, value, step=0, training=True):
    writer = self._get_writer(training)
    with writer.as_default():
      tf.summary.scalar(tag, value, step=step)

  def histogram(self, tag, values, step=0, training=True):
    writer = self._get_writer(training)
    with writer.as_default():
      tf.summary.histogram(tag, values, step=step)

  def image(self, tag, values, step=0, training=True):
    writer = self._get_writer(training)
    with writer.as_default():
      tf.summary.image(tag, data=values, step=step, max_outputs=values.shape[0])

  def profiler_trace(self):
    tf.summary.trace_on(graph=True, profiler=True)

  def profiler_export(self):
    tf.summary.trace_export(name='models', profiler_outdir=self._profiler_dir)

  def plot_traces(self,
                  tag,
                  signals,
                  spikes=None,
                  indexes=None,
                  step=0,
                  training=True):
    images = []

    if tf.is_tensor(signals):
      signals = signals.numpy()
    if len(signals.shape) > 2:
      signals = signals[0]

    signals = utils.set_array_format(
        signals, data_format='CW', hparams=self._hparams)

    # deconvolve signals if spikes aren't provided
    if spikes is None:
      spikes = spike_helper.deconvolve_signals(signals)

    if tf.is_tensor(spikes):
      spikes = spikes.numpy()
    if len(spikes.shape) > 2:
      spikes = spikes[0]

    spikes = utils.set_array_format(
        spikes, data_format='CW', hparams=self._hparams)

    if not indexes:
      indexes = range(len(signals))

    for i in indexes:
      image = self._plot_trace(signals[i], spikes[i], neuron=i)
      images.append(image)

    self.image(tag, values=tf.stack(images), step=step, training=training)

  def plot_histogram(self,
                     tag,
                     data,
                     xlabel=None,
                     ylabel=None,
                     title=None,
                     step=0,
                     training=False):
    assert type(data) == tuple
    images = []

    hist_kws = {
        "alpha": 0.6,
        "range":
        [min(min(data[0]), min(data[1])),
         max(max(data[0]), max(data[0]))]
    }

    ax = sns.distplot(
        data[0],
        bins=20,
        kde=False,
        hist_kws=hist_kws,
        color=self.real_color,
        label="Real")
    ax = sns.distplot(
        data[1],
        bins=20,
        kde=False,
        hist_kws=hist_kws,
        color=self.fake_color,
        label="Fake")

    ax.legend()

    if xlabel and ylabel and title:
      ax.set(xlabel=xlabel, ylabel=ylabel, title=title)

    plt.tight_layout()
    images.append(self._plot_to_image())
    plt.close()

    self.image(tag, values=tf.stack(images), step=step, training=training)

  def plot_histograms_grid(self,
                           tag,
                           data,
                           xlabel=None,
                           ylabel=None,
                           titles=None,
                           step=0,
                           training=False):
    assert type(data) == list and type(data[0]) == tuple
    images = []

    f, axes = plt.subplots(3, 3, figsize=(15, 15))
    i, row, col = 0, 0, 0
    while i < 9:
      real, fake = data[i]

      hist_kws = {
          "alpha": 0.6,
          "range": [min(min(real), min(fake)),
                    max(max(real), max(fake))]
      }

      ax = sns.distplot(
          real,
          bins=20,
          kde=False,
          hist_kws=hist_kws,
          color=self.real_color,
          label="Real",
          ax=axes[row, col])
      ax = sns.distplot(
          fake,
          bins=20,
          kde=False,
          hist_kws=hist_kws,
          color=self.fake_color,
          label="Fake",
          ax=axes[row, col])

      ax.legend()

      if xlabel and ylabel and titles:
        ax.set(xlabel=xlabel, ylabel=ylabel, title=titles[i])

      col += 1
      if col == 3:
        row += 1
        col = 0
      i += 1

    plt.tight_layout()
    images.append(self._plot_to_image())
    plt.close()

    self.image(tag, values=tf.stack(images), step=step, training=training)

  def plot_heatmaps(self,
                    tag,
                    matrix,
                    xlabel=None,
                    ylabel=None,
                    xticklabels='auto',
                    yticklabels='auto',
                    titles=None,
                    step=0,
                    training=False):
    assert type(matrix) == list and type(matrix[0]) == np.ndarray
    images = []
    for i in range(len(matrix)):
      f, ax = plt.subplots(figsize=(8, 8))
      ax = sns.heatmap(
          matrix[i],
          cmap='YlOrRd',
          xticklabels=xticklabels[i] if type(xticklabels) == list else 'auto',
          yticklabels=yticklabels[i] if type(xticklabels) == list else 'auto',
          ax=ax)
      if xlabel is not None and ylabel is not None:
        ax.set(xlabel=xlabel, ylabel=ylabel)
      if titles is not None:
        ax.set_title(titles[i])
      image = self._plot_to_image()
      plt.close()
      images.append(image)
    self.image(tag, values=tf.stack(images), step=step, training=training)

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

  def raster_plot(self,
                  tag,
                  real_spikes,
                  fake_spikes,
                  xlabel=None,
                  ylabel=None,
                  title=None,
                  step=0,
                  training=True):

    real_x, real_y = np.nonzero(real_spikes)
    fake_x, fake_y = np.nonzero(fake_spikes)

    df = pd.DataFrame({
        'x': np.concatenate([real_y, fake_y]),
        'y': np.concatenate([real_x, fake_x]),
        'real_data': [True] * len(real_x) + [False] * len(fake_x)
    })

    g = sns.JointGrid(x='x', y='y', data=df)
    plt.gcf().set_size_inches(16, 10)

    real = df.loc[df.real_data == True]
    fake = df.loc[df.real_data == False]

    sns.scatterplot(
        real.x,
        real.y,
        marker='|',
        color=self.real_color,
        alpha=0.8,
        ax=g.ax_joint)
    ax = sns.scatterplot(
        fake.x,
        fake.y,
        marker='|',
        color=self.fake_color,
        alpha=0.8,
        ax=g.ax_joint)
    ax.set(xlabel=xlabel, ylabel=ylabel)

    sns.distplot(
        real.x, kde=False, color=self.real_color, ax=g.ax_marg_x, bins=40)
    ax = sns.distplot(
        fake.x, kde=False, color=self.fake_color, ax=g.ax_marg_x, bins=40)
    ax.set_xlabel('')

    sns.distplot(
        real.y,
        kde=False,
        color=self.real_color,
        ax=g.ax_marg_y,
        bins=40,
        vertical=True)
    ax = sns.distplot(
        fake.y,
        kde=False,
        color=self.fake_color,
        ax=g.ax_marg_y,
        bins=40,
        vertical=True)
    ax.set_ylabel('')

    g.ax_joint.legend(
        labels=['real', 'fake'],
        ncol=2,
        frameon=False,
        prop={
            'weight': 'regular',
            'size': 12
        },
        loc='upper right')

    image = self._plot_to_image()
    plt.close()
    images = tf.expand_dims(image, axis=0)

    self.image(tag, values=images, step=step, training=training)

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
