import os
import pickle
import numpy as np
import tensorflow as tf


def get_dataset(hparams):
  if not os.path.exists(hparams.input):
    print('input pickle {} cannot be found'.format(hparams.input))
    exit()

  with open(hparams.input, 'rb') as file:
    segments = pickle.load(file)

  np.random.shuffle(segments)

  hparams.sequence_length = segments.shape[-1]

  # 70% training data
  train_size = int(len(segments) * 0.7)

  # train set
  train_ds = tf.data.Dataset.from_tensor_slices(segments[:train_size])
  train_ds = train_ds.shuffle(buffer_size=512)
  train_ds = train_ds.batch(hparams.batch_size)
  train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

  # validation set
  validation_ds = tf.data.Dataset.from_tensor_slices(segments[train_size:])
  validation_ds = validation_ds.batch(hparams.batch_size)

  hparams.steps_per_epoch = int(np.ceil(train_size / hparams.batch_size))

  return train_ds, validation_ds
