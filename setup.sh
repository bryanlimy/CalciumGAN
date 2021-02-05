#!/bin/sh

macOS=false
current_dir="$(pwd -P)"

check_requirements() {
  case "$(uname -s)" in
    Darwin)
      printf 'Installing on macOS...'
      export CFLAGS='-stdlib=libc++'
      macOS=true
      ;;
    Linux)
      printf 'Installing on Linux...'
      ;;
    *)
      printf 'The installation script only support Linux and macOS.'
      exit 1
      ;;
  esac
}

install_python_packages() {
  printf '\nInstalling tensorflow...\n'
  if [ $macOS = "true" ]; then
    pip install -q tensorflow==2.4.1
  else
    conda install -q -c nvidia cudatoolkit=11.0 cudnn=8.0 nccl -y
    pip install -q tensorflow==2.4.1
  fi
  printf '\nInstalling other Python packages...\n'
  pip install -q -r requirements.txt
}

set_python_path() {
  printf "\nSet conda environment variables...\n"
  conda env config vars set PYTHONPATH=$PYTHONPATH:$current_dir
}


check_requirements
install_python_packages
set_python_path

printf '\nSetup completed.'
