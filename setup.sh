#!/bin/sh

current_dir="$(pwd)"
macOS=false

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
      printf 'Only Linux and macOS systems are currently supported.'
      exit 1
      ;;
  esac
}

install_python_packages() {
  printf '\nInstall tensorflow\n'
  if [ "$macOS" = "true" ]; then
    python3 -m pip install tensorflow==2.3.1
  else
    conda install cudatoolkit=10.1 cudnn=7.6 cupti=10.1 numpy blas scipy -y
    python3 -m pip install tensorflow==2.3.1
  fi
  printf '\nInstall Python packages\n'
  python3 -m pip install -r requirements.txt
}

install_oasis() {
  printf '\nInstalling OASIS...'
  cd "$current_dir" || exit 1
  printf '\nInstalling Gurobi...'
  conda config --add channels http://conda.anaconda.org/gurobi
  conda install gurobi -y
  printf '\nInstalling Mosek...'
  conda install -c mosek mosek -y
  git clone https://github.com/j-friedrich/OASIS.git oasis
  cd oasis || exit 1
  python3 setup.py build_ext --inplace
  python3 -m pip install -e .
}

install_elephant() {
  printf '\nInstalling Elephant...'
  cd "$current_dir" || exit 1
  git clone git://github.com/NeuralEnsemble/elephant.git elephant
  cd elephant || exit 1
  python3 setup.py install
}

check_requirements
install_python_packages
install_oasis
install_elephant

printf '\nSetup completed.'
