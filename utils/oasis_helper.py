import os
import h5py
import numpy as np
import tensorflow as tf
from oasis.functions import deconvolve
from multiprocessing import Process, Queue, cpu_count


def split(sequence, n):
  """ divide sequence into n sub-sequence evenly"""
  k, m = divmod(len(sequence), n)
  return [
      sequence[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)
  ]


def _deconvolve_signals(signals, queue=None):
  spikes = []
  for i in range(len(signals)):
    c, s, b, g, lam = deconvolve(signals[i], g=(None,), penalty=1)
    spikes.append(s / s.max() if s.max() > 0 else s)
    if queue is not None and len(spikes) >= 500:
      queue.put(np.array(spikes, dtype=np.float32))
      spikes = []
  if queue is None:
    return spikes
  elif len(spikes) > 0:
    queue.put(np.array(spikes, dtype=np.float32))


def _append_h5(ds, value):
  """ append value to a H5 dataset ds """
  if type(value) != np.ndarray:
    value = np.array(value, dtype=np.float32)
  ds.resize((ds.shape[0] + value.shape[0]), axis=0)
  ds[-value.shape[0]:] = value


def _async_writer(filename, queue):
  with h5py.File(filename, 'w') as output:
    while True:
      value = queue.get()
      if value is None:
        break
      if 'spikes' in output:
        _append_h5(output['spikes'], value)
      else:
        output.create_dataset(
            'spikes',
            dtype=np.float32,
            data=value,
            chunks=True,
            maxshape=(None, value.shape[1]))


def deconvolve_signals(hparams, signals, to_tensor=False, multiprocessing=True):
  if tf.is_tensor(signals):
    signals = signals.numpy()

  signals = signals.astype('double')

  if multiprocessing:
    num_jobs = min(len(signals), cpu_count() - 4)
    subsets = split(signals, n=num_jobs)

    # cache h5 file to store spike trains
    cache = os.path.join(hparams.output_dir, 'cache.h5')

    queue = Queue()
    writer = Process(target=_async_writer, args=(cache, queue))
    writer.start()

    jobs = []
    for i in range(num_jobs):
      job = Process(target=_deconvolve_signals, args=(subsets[i], queue))
      jobs.append(job)
      job.start()

    for job in jobs:
      job.join()

    queue.put(None)
    writer.join()

    with h5py.File(cache, 'r') as h5:
      spikes = np.array(h5['spikes'][:], dtype=np.float32)

    os.remove(cache)
  else:
    spikes = np.array(_deconvolve_signals(signals))

  assert spikes.shape == signals.shape

  return tf.convert_to_tensor(spikes, dtype=tf.float32) if to_tensor else spikes
