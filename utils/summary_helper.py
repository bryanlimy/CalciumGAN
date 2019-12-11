import os
import io
import numpy as np
import tensorflow as tf
from .oasis_helper import deconvolve_signals

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Summary(object):
  """ 
  Log tf.Summary to output_dir during training and output_dir/eval during 
  evaluation
  """

  def __init__(self, hparams):
    self._hparams = hparams
    self.train_writer = tf.summary.create_file_writer(hparams.output_dir)
    self.val_writer = tf.summary.create_file_writer(
        os.path.join(hparams.output_dir, 'validation'))
    tf.summary.trace_on(graph=True, profiler=False)

  def _get_writer(self, training):
    return self.train_writer if training else self.val_writer

  def _get_step(self):
    return self._hparams.global_step

  def _plot_to_image(self):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    buf = io.BytesIO()
    plt.savefig(buf, dpi=100, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    return image

  def _simple_axis(self, axis):
    """plot only x and y axis, not a frame for subplot ax"""
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.get_xaxis().tick_bottom()
    axis.get_yaxis().tick_left()

  def _plot_trace(self, signal, spike):
    plt.figure(figsize=(20, 4))
    plt.subplot(211)
    plt.plot(signal, label='signal', zorder=-12, c='r')
    plt.legend(ncol=3, frameon=False, loc=(.02, .85))
    self._simple_axis(plt.gca())
    plt.tight_layout()
    # plot spike train
    plt.subplot(212)
    plt.bar(np.arange(len(spike)), spike, width=0.3, label='oasis', color='b')
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
    signals = signals[0]

    if spikes is None:
      spikes = deconvolve_signals(signals, multiprocessing=False)
    if tf.is_tensor(spikes):
      spikes = spikes.numpy()

    # plot 20 neurons at most
    for i in range(min(20, signals.shape[0])):
      image = self._plot_trace(signals[i], spikes[i])
      images.append(image)
    images = tf.stack(images)
    self.image(tag, values=images, step=step, training=training)

  def graph(self):
    writer = self._get_writer(training=True)
    with writer.as_default():
      tf.summary.trace_export(name='models', step=0)
