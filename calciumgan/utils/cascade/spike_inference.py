import os
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model

from calciumgan.utils import h5_helper as h5
from calciumgan.utils.cascade.cascade2p import cascade, config

tf.get_logger().setLevel('ERROR')


def spike_inference(signals_filename,
                    spikes_filename,
                    model='Universal_25Hz_smoothing100ms'):
  if not os.path.exists(signals_filename):
    print('{} not found'.format(signals_filename))
    exit()

  if os.path.exists(spikes_filename):
    os.remove(spikes_filename)

  tf.keras.backend.clear_session()

  with tf.device('/CPU:0'):
    cascade.download_model(model)

    model_path = os.path.join(
        os.path.dirname(__file__), 'pretrained_models', model)
    model_config = config.read_config(os.path.join(model_path, 'config.yaml'))
    model_dict = cascade.get_model_paths(model_path)
    models = {
        noise_level: [load_model(model_path) for model_path in model_paths
                     ] for noise_level, model_paths in model_dict.items()
    }

    for key in h5.get_keys(signals_filename):
      print(f'deconvolve {key} from {signals_filename}...')
      signals = h5.get(signals_filename, key=key)

      # convert traces to NCW (num. samples, num. neurons, time steps)
      signals = np.transpose(signals, axes=[0, 2, 1])
      spike_rates = np.zeros(shape=signals.shape, dtype=np.float32)
      for i in tqdm(range(signals.shape[0]), desc=f'deconvolve {key}'):
        spike_rates[i] = cascade.predict(
            signals[i],
            models=models,
            batch_size=model_config['batch_size'],
            sampling_rate=model_config['sampling_rate'],
            before_frac=model_config['before_frac'],
            window_size=model_config['windowsize'],
            noise_levels_model=model_config['noise_levels'],
            smoothing=model_config['smoothing'],
        )
      # convert spikes to NWC (num. samples, time steps, num. neurons)
      spike_rates = np.transpose(spike_rates, axes=[0, 2, 1])
      h5.write(spikes_filename, data={key: spike_rates})


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--signals_filename', type=str, required=True)
  parser.add_argument('--spikes_filename', type=str, required=True)
  args = parser.parse_args()
  spike_inference(
      signals_filename=args.signals_filename,
      spikes_filename=args.spikes_filename)
