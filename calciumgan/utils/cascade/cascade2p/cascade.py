# Code from https://github.com/HelmchenLabSoftware/Cascade/blob/master/cascade2p/cascade.py

# -*- coding: utf-8 -*-
"""  High level interface to the CASCADE package

This file contains functions to train networks for spike prediction ('train_model')
and to use existing networks to predict spiking activity ('predict').


A typical workflow for applying an existing network to calcium imaging data,
shown in the "demo_predict.py" script:

  1)  Load calcium imaging data as a dF/F matrix
  2)  Load a predefined model; the model should match the properties of the calcium
      imaging dataset (frame rate, noise levels, ground truth datasets)
  3)  Use the model and the dF/F matrix as inputs for the function 'predict'
  4)  Predictions will be saved. Done!

A typical workflow for training a new network would be the following,
shown in the "demo_train.py" script:

  1)  Define a model (frame rate, noise levels, ground truth datasets; additional parameters)
  2)  Use the model as input to the function 'train_model'
  3)  The trained models will be saved together with a configuration file (YAML). Done!


Additional functions in this file are used to navigate different models 
('get_model_paths', 'create_model_folder',  'verify_config_dict').

"""

import os
import re
import glob
import time
import zipfile
import warnings
import numpy as np
import tensorflow.keras as keras
from urllib.request import urlopen
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation

from . import config, utils


def train_model(model_name,
                model_folder='pretrained_models',
                ground_truth_folder='Ground_truth'):
  """ 
  Train neural network with parameters specified in the config.yaml file in the model folder

  In this function, a model is configured (defined in the input 'model_name': frame rate, noise levels, ground truth datasets, etc.).
  The ground truth is resampled (function 'preprocess_groundtruth_artificial_noise_balanced', defined in "utils.py").
  The network architecture is defined (function 'define_model', defined in "utils.py").
  The thereby defined model is trained with the resampled ground truth data.
  The trained model with its weight and configuration details is saved to disk.

  Parameters
  ----------
  model_name : str
    Name of the model, e.g. 'Universal_30Hz_smoothing100ms'
    This name has to correspond to the folder with the config.yaml file which defines the model parameters

  model_folder: str
    Absolute or relative path, which defines the location of the specified model_name folder
    Default value 'pretrained_models' assumes a current working directory in the Cascade folder

  ground_truth_folder : str
    Absolute or relative path, which defines the location of the ground truth datasets
    Default value 'Ground_truth'  assumes a current working directory in the Cascade folder

  Returns
  --------
  None
    All results are saved in the folder model_name as .h5 files containing the trained model
  """
  model_path = os.path.join(model_folder, model_name)
  cfg_file = os.path.join(model_path, 'config.yaml')

  # check if configuration file can be found
  if not os.path.isfile(cfg_file):
    m = 'The configuration file "config.yaml" can not be found at the location "{}".\n'.format( os.path.abspath(cfg_file) ) + \
        'You have provided the model "{}" at the absolute or relative path "{}".\n'.format( model_name, model_folder) + \
        'Please check if there is a folder for model "{}" at the location "{}".'.format( model_name, os.path.abspath(model_folder))
    print(m)
    raise Exception(m)

  # load cfg dictionary from config.yaml file
  cfg = config.read_config(cfg_file)
  verbose = cfg['verbose']

  if verbose:
    print('Used configuration for model fitting (file {}):\n'.format(
        os.path.abspath(cfg_file)))
    for key in cfg:
      print('{}:\t{}'.format(key, cfg[key]))

    print('\n\nModels will be saved into this folder:',
          os.path.abspath(model_path))

  # add base folder to selected training datasets
  training_folders = [
      os.path.join(ground_truth_folder, ds) for ds in cfg['training_datasets']
  ]

  # check if the training datasets can be found
  missing = False
  for folder in training_folders:
    if not os.path.isdir(folder):
      print('The folder "{}" could not be found at the specified location "{}"'.
            format(folder, os.path.abspath(folder)))
      missing = True
  if missing:
    m = 'At least one training dataset could not be located.\nThis could mean that the given path "{}" '.format(ground_truth_folder) + \
        'does not specify the correct location or that e.g. a training dataset referenced in the config.yaml file ' + \
        'contained a typo.'
    print(m)
    raise Exception(m)

  start = time.time()
  # Update model fitting status
  cfg['training_finished'] = 'Running'
  config.write_config(cfg, os.path.join(model_path, 'config.yaml'))

  nr_model_fits = len(cfg['noise_levels']) * cfg['ensemble_size']
  print('Fitting a total of {} models:'.format(nr_model_fits))

  curr_model_nr = 0

  print(training_folders[0])

  for noise_level in cfg['noise_levels']:
    for ensemble in range(cfg['ensemble_size']):
      # train 'ensemble_size' (e.g. 5) models for each noise level

      curr_model_nr += 1
      print(
          '\nFitting model {} with noise level {} (total {} out of {}).'.format(
              ensemble + 1, noise_level, curr_model_nr, nr_model_fits))

      # preprocess dataset to get uniform dataset for training
      X, Y = utils.preprocess_groundtruth_artificial_noise_balanced(
          ground_truth_folders=training_folders,
          before_frac=cfg['before_frac'],
          windowsize=cfg['windowsize'],
          after_frac=1 - cfg['before_frac'],
          noise_level=noise_level,
          sampling_rate=cfg['sampling_rate'],
          smoothing=cfg['smoothing'] * cfg['sampling_rate'],
          omission_list=[],
          permute=1,
          verbose=cfg['verbose'],
          replicas=1,
          causal_kernel=cfg['causal_kernel'])

      model = utils.define_model(
          filter_sizes=cfg['filter_sizes'],
          filter_numbers=cfg['filter_numbers'],
          dense_expansion=cfg['dense_expansion'],
          windowsize=cfg['windowsize'],
          loss_function=cfg['loss_function'],
          optimizer=cfg['optimizer'])

      model.compile(loss=cfg['loss_function'], optimizer=cfg['optimizer'])

      model.fit(
          X,
          Y,
          batch_size=cfg['batch_size'],
          epochs=cfg['nr_of_epochs'],
          verbose=cfg['verbose'])

      # save model
      file_name = 'Model_NoiseLevel_{}_Ensemble_{}.h5'.format(
          int(noise_level), ensemble)
      model.save(os.path.join(model_path, file_name))
      print('Saved model:', file_name)

  # Update model fitting status
  cfg['training_finished'] = 'Yes'
  config.write_config(cfg, os.path.join(model_path, 'config.yaml'))

  print('\n\nDone!')
  print('Runtime: {:.0f} min'.format((time.time() - start) / 60))


def predict(traces,
            models,
            batch_size,
            sampling_rate,
            before_frac,
            window_size,
            noise_levels_model,
            smoothing,
            threshold=0,
            padding=np.nan):
  # calculate noise levels for each trace
  trace_noise_levels = utils.calculate_noise_levels(traces, sampling_rate)

  # XX has shape: (neurons, timepoints, windowsize)
  XX = utils.preprocess_traces(traces, before_frac, window_size)
  Y_predict = np.zeros((XX.shape[0], XX.shape[1]), dtype=np.float32)

  # Use for each noise level the matching model
  for i, model_noise in enumerate(noise_levels_model):
    # select neurons which have this noise level:
    if i == 0:
      # lowest noise
      neuron_idx = np.where(trace_noise_levels < model_noise + 0.5)[0]
    elif i == len(noise_levels_model) - 1:
      # highest noise
      neuron_idx = np.where(trace_noise_levels >= model_noise - 0.5)[0]
    else:
      neuron_idx = np.where((trace_noise_levels >= model_noise - 0.5) &
                            (trace_noise_levels < model_noise + 0.5))[0]
    if len(neuron_idx) == 0:
      continue

    # select neurons and merge neurons and timepoints into one dimension
    XX_sel = XX[neuron_idx, :, :]
    XX_sel = np.reshape(XX_sel,
                        (XX_sel.shape[0] * XX_sel.shape[1], XX_sel.shape[2]))
    # add empty third dimension to match training shape
    XX_sel = np.expand_dims(XX_sel, axis=2)

    for model in models[model_noise]:
      prediction_flat = model.predict(XX_sel, batch_size=batch_size)
      prediction = np.reshape(prediction_flat, (len(neuron_idx), XX.shape[1]))
      # average predictions
      Y_predict[neuron_idx, :] += prediction / len(models)

    # remove models from memory
    keras.backend.clear_session()

  if threshold == 0:
    # ignore warning because of nan's in Y_predict in comparison with value
    with np.errstate(invalid='ignore'):
      Y_predict[Y_predict < 0] = 0
  if threshold == 1:
    # Cut off noise floor (lower than 1/e of a single action potential)
    # find out empirically  how large a single AP is (depends on frame rate
    # and smoothing)
    single_spike = np.zeros(1001,)
    single_spike[501] = 1
    single_spike_smoothed = gaussian_filter(
        single_spike.astype(float), sigma=smoothing * sampling_rate)
    threshold_value = np.max(single_spike_smoothed) / np.exp(1)

    # Set everything below threshold to zero.
    # Use binary dilation to avoid clipping of true events.
    for neuron in range(Y_predict.shape[0]):
      # ignore warning because of nan's in Y_predict in comparison with value
      with np.errstate(invalid='ignore'):
        activity_mask = Y_predict[neuron, :] > threshold_value
      activity_mask = binary_dilation(
          activity_mask, iterations=int(smoothing * sampling_rate))
      Y_predict[neuron, ~activity_mask] = 0
      # set possible negative values in dilated mask to 0
      Y_predict[Y_predict < 0] = 0

  # NaN or 0 for first and last datapoints, for which no predictions can be made
  Y_predict[:, 0:int(before_frac * window_size)] = padding
  Y_predict[:, -int((1 - before_frac) * window_size):] = padding

  return Y_predict.astype(np.float32)


def verify_config_dict(config_dictionary):
  """ Perform some test to catch the most likely errors when creating config files """

  # TODO: Implement
  print('Not implemented yet...')


def create_model_folder(config_dictionary, model_folder='pretrained_models'):
  """ Creates a new folder in model_folder and saves config.yaml file there

    Parameters
    ----------
    config_dictionary : dict
        Dictionary with keys like 'model_name' or 'training_datasets'
        Values which are not specified will be set to default values defined in
        the config_template in config.py

    model_folder : str
        Absolute or relative path, which defines the location at which the new
        folder containing the config file will be created
        Default value 'pretrained_models' assumes a current working directory
        in the Cascade folder

    """
  cfg = config_dictionary  # shorter name

  # TODO: call here verify_config_dict

  # TODO: check here the current directory, might not be the main folder...
  model_path = os.path.join(model_folder, cfg['model_name'])

  if not os.path.exists(model_path):
    # create folder
    try:
      os.mkdir(model_path)
      print('Created new directory "{}"'.format(os.path.abspath(model_path)))
    except:
      print(model_path + ' already exists')

    # save config file into the folder
    config.write_config(cfg, os.path.join(model_path, 'config.yaml'))

  else:
    warnings.warn('There is already a folder called {}. '.format(cfg['model_name']) + \
          'Please rename your model.')


def get_model_paths(model_path):
  """
  Find all models in the model folder and return as dictionary
  ( Helper function called by predict() )

  Returns
  -------
  model_dict : dict
      Dictionary with noise_level (int) as keys and entries are lists of model paths
  """

  all_models = glob.glob(os.path.join(model_path, '*.h5'))
  all_models = sorted(all_models)  # sort

  # Exception in case no model was found to catch this mistake where it happened
  if len(all_models) == 0:
    m = 'No models (*.h5 files) were found in the specified folder "{}".'.format(
        os.path.abspath(model_path))
    raise Exception(m)

  # dictionary with key for noise level, entries are lists of models
  model_dict = dict()

  for model_path in all_models:
    try:
      noise_level = int(re.findall('_NoiseLevel_(\d+)', model_path)[0])
    except:
      print('Error while processing the file with name: ', model_path)
      raise

    # add model path to the model dictionary
    if noise_level not in model_dict:
      model_dict[noise_level] = list()
    model_dict[noise_level].append(model_path)

  return model_dict


def download_model(
    model_name,
    model_folder='calciumgan/utils/cascade/pretrained_models',
    info_file_link='https://drive.switch.ch/index.php/s/lBJd2JfVrkmSSiN/download',
    overwrite=False,
    verbose=0):
  """
  Download and unzip pretrained model from the online repository
 
  Parameters
  ----------
  model_name : str
    Name of the model, e.g. 'Universal_30Hz_smoothing100ms'
    This name has to correspond to a pretrained model that is available for 
    download. To see available models, run this function with 
    model_name='update_models' and check the downloaded file 
    'available_models.yaml'

  model_folder: str
    Absolute or relative path, which defines the location of the specified 
    model_name folder
    Default value 'pretrained_models' assumes a current working directory in 
    the Cascade folder

  info_file_link: str
    Direct download link to yaml file which contains download links for new models.
    Default value is official repository of models.

  overwrite: bool
    Overwrite the existing model if it exists.

  verbose : int
    If 0, no messages are printed. if larger than 0, the user is informed about status.
  """
  if os.path.exists(os.path.join(model_folder, model_name)) and not overwrite:
    return

  # Download the current yaml file with information about available models first
  new_file = os.path.join(model_folder, 'available_models.yaml')
  with urlopen(info_file_link) as response:
    text = response.read()

  with open(new_file, 'wb') as f:
    f.write(text)

  # check if the specified model_name is present
  # orderedDict with model names as keys
  download_config = config.read_config(new_file)

  if model_name not in download_config.keys():
    if model_name == 'update_models':
      print(f'You can now check the updated available_models.yaml file for '
            f'valid model names.\nFile location: {os.path.abspath(new_file)}')
      return
    raise Exception(f'The specified model_name "{model_name}" is not in the '
                    f'list of available models. Available models for download '
                    f'are: {list(download_config.keys())}')

  if verbose:
    print(f'Downloading and extracting new model "{model_name}"...')

  # download and save .zip file of model
  download_link = download_config[model_name]['Link']
  with urlopen(download_link) as response:
    data = response.read()

  tmp_file = os.path.join(model_folder, 'tmp_zipped_model.zip')
  with open(tmp_file, 'wb') as f:
    f.write(data)

  # unzip the model and save in the corresponding folder
  with zipfile.ZipFile(tmp_file, 'r') as zip_ref:
    zip_ref.extractall(path=model_folder)

  os.remove(tmp_file)

  if verbose:
    print('Pretrained model was saved in folder "{}"'.format(
        os.path.abspath(os.path.join(model_folder, model_name))))

  return
