import os
import io
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

from . import spike_helper


class Summary(object):
  """ 
  Log tf.Summary to output_dir during training and output_dir/eval during 
  evaluation
  """

  def __init__(self, hparams, policy=None):
    self._hparams = hparams
    self.train_writer = tf.summary.create_file_writer(hparams.output_dir)
    self.val_writer = tf.summary.create_file_writer(
        os.path.join(hparams.output_dir, 'validation'))
    tf.summary.trace_on(graph=True, profiler=False)
    self._policy = policy
    # color for matplotlib
    self._real_color = 'dodgerblue'
    self._fake_color = 'orangered'

  def _get_writer(self, training):
    return self.train_writer if training else self.val_writer

  def _get_step(self):
    return self._hparams.global_step

  def _get_loss_scale(self):
    if self._policy is None:
      return None
    else:
      return self._policy.loss_scale._current_loss_scale

  def _plot_to_image(self):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    buf = io.BytesIO()
    plt.savefig(buf, dpi=100, format='png')
    buf.seek(0)
    return tf.image.decode_png(buf.getvalue(), channels=4)

  def _simple_axis(self, axis):
    """plot only x and y axis, not a frame for subplot ax"""
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.get_xaxis().tick_bottom()
    axis.get_yaxis().tick_left()

  def _plot_trace(self, signal, spike):
    plt.figure(figsize=(20, 4))
    # plot signal
    plt.subplot(211)
    plt.plot(signal, label='signal', zorder=-12, color=self._real_color)
    plt.legend(ncol=3, frameon=False, loc=(.02, .85))
    self._simple_axis(plt.gca())
    plt.tight_layout()
    # plot spike train
    plt.subplot(212)
    plt.bar(
        range(len(spike)),
        spike,
        width=0.4,
        label='spike',
        color=self._fake_color)
    plt.ylim(0, 1.3)
    plt.legend(ncol=3, frameon=False, loc=(.02, .85))
    self._simple_axis(plt.gca())
    plt.tight_layout()
    image = self._plot_to_image()
    plt.close()
    return image

  def scalar(self, tag, value, step=None, training=True):
    writer = self._get_writer(training)
    step = self._get_step() if step is None else step
    with writer.as_default():
      tf.summary.scalar(tag, value, step=step)

  def histogram(self, tag, values, step=None, training=True):
    writer = self._get_writer(training)
    step = self._get_step() if step is None else step
    with writer.as_default():
      tf.summary.histogram(tag, values, step=step)

  def image(self, tag, values, step=None, training=True):
    writer = self._get_writer(training)
    step = self._get_step() if step is None else step
    with writer.as_default():
      tf.summary.image(tag, data=values, step=step, max_outputs=values.shape[0])

  def plot_traces(self, tag, signals, spikes=None, step=None, training=True):
    images = []

    if tf.is_tensor(signals):
      signals = signals.numpy()
    if len(signals.shape) > 2:
      signals = signals[0]

    # deconvolve signals if spikes aren't provided
    if spikes is None:
      spikes = spike_helper.deconvolve_signals(signals)

    if tf.is_tensor(spikes):
      spikes = spikes.numpy()
    if len(spikes.shape) > 2:
      spikes = spikes[0]

    # plot traces at most
    for i in range(min(20, signals.shape[0])):
      image = self._plot_trace(signals[i], spikes[i])
      images.append(image)
    self.image(tag, values=tf.stack(images), step=step, training=training)

  def plot_histogram(self,
                     tag,
                     data,
                     xlabel=None,
                     ylabel=None,
                     step=None,
                     training=False):
    plt.hist(
        data,
        bins=20,
        label=['real', 'fake'],
        color=[self._real_color, self._fake_color],
        alpha=0.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    image = self._plot_to_image()
    plt.close()
    self.image(tag, values=tf.stack([image]), step=step, training=training)

  def graph(self):
    writer = self._get_writer(training=True)
    with writer.as_default():
      tf.summary.trace_export(name='models', step=0)

  def variable_summary(self, variable, name=None, step=None, training=True):
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

  def plot_weights(self, gan, step=None, training=True):
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
          training=True):
    self.scalar('loss/generator', gen_loss, training=training)
    self.scalar('loss/discriminator', dis_loss, training=training)
    if gradient_penalty is not None:
      self.scalar('loss/gradient_penalty', gradient_penalty, training=training)
    if metrics is not None:
      for tag, value in metrics.items():
        self.scalar(tag, value, training=training)
    if elapse is not None:
      self.scalar('elapse', elapse, training=training)
    if gan is not None and self._hparams.plot_weights:
      self.plot_weights(gan, training=training)
    if not training and self._policy is not None:
      self.scalar('model/loss_scale', self._get_loss_scale(), training=training)
