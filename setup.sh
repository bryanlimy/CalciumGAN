#!/bin/sh

use_gpu=false

print_usage() {
  echo 'Usage: sh setup.sh --use_gpu --package_manager'
  echo '---gpu,-g                     install tensorflow-gpu'
}

check_requirements() {
  case "$(uname -s)" in
    Darwin)
      echo 'Installing on macOS.'
      export CFLAGS='-stdlib=libc++'
      ;;
    Linux)
      echo 'Install on Linux.'
      ;;
    *)
      echo 'Only Linux and macOS systems are currently supported.'
      exit 1
      ;;
  esac
}

install_python_packages() {
  cd "$(dirname "$0")"
  if [ "$use_gpu" = "true" ]; then
    echo '\nInstall tensorflow-gpu'
    python3 -m pip install tensorflow-gpu==2.0.0
  else
    echo '\nInstall tensorflow'
    python3 -m pip install tensorflow==2.0.0
  fi
  echo '\nInstall Python packages'
  python3 -m pip install -r requirements.txt
}

install_oasis() {
  echo '\n\nInstall OASIS'
  echo '\nInstall Gurobi'
  conda config --add channels http://conda.anaconda.org/gurobi
  conda install -c gurobi -y
  echo '\nInstall Mosek'
  conda install -c mosek mosek -y
  git clone https://github.com/j-friedrich/OASIS.git oasis
  cd oasis

  python3 setup.py build_ext --inplace
  python3 -m pip install -e .
}


# Read flags and arguments
while [ ! $# -eq 0 ]; do
  case "$1" in
    --help | -h)
      print_usage
      exit 1
      ;;
    --gpu | -g)
      use_gpu=true
      ;;
    *)
      echo "Unknown flag $1, please check available flags with --help"
      exit 1
      ;;
  esac
  shift
done

check_requirements
install_python_packages
install_oasis

echo '\nSetup completed.'
