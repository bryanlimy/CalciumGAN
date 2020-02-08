#!/bin/bash

function print_help() {
  echo 'Usage: ./run_docker.sh [OPTIONS]'
  echo 'Options:'
  echo '  --gpu                     use GPUs'
}

use_gpu=false

while [ ! $# -eq 0 ]; do
  case "$1" in
    --help | -h)
      print_help
      exit 1
      ;;
    --gpu)
      use_gpu=true
      ;;
    *)
      echo "Unknown flag $1, please check available flags with --help"
      exit 1
      ;;
  esac
  shift
done

function main() {
  if [ "$use_gpu" = "true" ]; then
    docker run --gpus all -it \
      -v ~/Git/calcium_imaging_gan/:/home/bryanlimy/calcium_imaging_gan \
      -v /media/data0/bryanlimy/calcium_datasets/:/home/bryanlimy/calcium_imaging_gan/dataset/tfrecords \
      -v /media/data0/bryanlimy/runs:/home/bryanlimy/calcium_imaging_gan/runs \
      bryanlimy/projects:0.2-calcium-gan-base zsh
  else
    docker run -it \
      -v ~/Git/calcium_imaging_gan/:/home/bryanlimy/calcium_imaging_gan \
      -v /media/data0/bryanlimy/calcium_datasets/:/home/bryanlimy/calcium_imaging_gan/dataset/tfrecords \
      -v /media/data0/bryanlimy/runs:/home/bryanlimy/calcium_imaging_gan/runs \
      bryanlimy/projects:0.2-calcium-gan-base zsh
  fi
}

main
