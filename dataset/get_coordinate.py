import os
import h5py
import pickle
import argparse
import numpy as np


def get_coordinate(filename):
  if not os.path.exists(filename):
    print('file {} does not exists'.format(filename))
    exit()

  coordinates = []
  with h5py.File(filename, 'r') as file:
    rois = file['data'].value
    for roi in rois[2:]:
      coordinates.append(file[roi[0]]['mnCoordinates'].value)

  print(coordinates)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--filename', default='raw_data/MC_20181117_P01.mat', type=str)
  arguments = parser.parse_args()
  get_coordinate(filename=arguments.filename)
