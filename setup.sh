#!/bin/sh

macOS=false
current_dir="$(pwd)"

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
  printf '\nInstalling tensorflow...'
  if [ "$macOS" = "true" ]; then
    pip install -q tensorflow==2.4.1
  else
    conda install -q -c nvidia cudatoolkit=11.0 cudnn=8.0 nccl -y
    pip install -q tensorflow==2.4.1
  fi
  printf '\nInstalling other Python packages...'
  pip install -q -r requirements.txt
}

set_python_path() {
  path='PYTHONPATH=$PYTHONPATH:'$current_dir
  case $SHELL in
    */zsh)
      printf "\nUpdating ~/.zshrc to set PYTHONPATH..."
      echo "# CalciumGAN PYTHONPATH\nexport $path\n" >> ~/.zshrc
      ;;
    */bash)
      printf "\nUpdating ~/.bashrc to set PYTHONPATH..."
      echo "# CalciumGAN PYTHONPATH\nexport $path\n" >> ~/.bashrc
      ;;
    *)
      ;;
  esac
  printf "\nPlease run export $path\n"
}

check_requirements
install_python_packages
set_python_path

printf '\nSetup completed.'
