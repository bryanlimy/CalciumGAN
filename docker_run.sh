#!/bin/bash

function print_help() {
  echo "Usage: ./run_docker.sh --name [NAME] [OPTIONS]"
  echo "  --name    name of the container"
  echo "  --cpus    number of CPU cores to use"
  echo "  --gpus    use GPUs"
  echo "  --root    run container as root user"
}

name='container_1'
cpus=8
gpus=false
as_root=false

while [ ! $# -eq 0 ]; do
  case "$1" in
    --help | -h)
      print_help
      exit 1
      ;;
    --name)
      shift
      name=$1
      ;;
    --cpus)
      shift
      cpus=$1
      ;;
    --gpus)
      gpus=true
      ;;
    --root)
      as_root=true
      ;;
    *)
      echo "Unknown flag $1, please check available flags with --help"
      exit 1
      ;;
  esac
  shift
done

function main() {
  command="docker run -it --rm"
  if [ "$as_root" = "false" ]; then
    command+=" "
    command+="-u $(id -u):$(id -g)"
  fi
  command+=" "
  command+="--name $name"
  command+=" "
  command+="--cpus $cpus"
  if [ "$gpus" = "true" ]; then
    command+=" "
    command+="--gpus all"
  fi
  command+=" "
  command+="-v ~/Git/calcium_imaging_gan/:/home/bryanlimy/calcium_imaging_gan"
  command+=" "
  command+="-v /media/data0/bryanlimy/calcium_datasets/:/home/bryanlimy/calcium_imaging_gan/dataset/tfrecords"
  command+=" "
  command+="-v /media/data0/bryanlimy/runs:/home/bryanlimy/calcium_imaging_gan/runs"
  command+=" "
  command+="bryanlimy/projects:0.6-calcium-gan-base zsh"
  eval "$command"
}

main
