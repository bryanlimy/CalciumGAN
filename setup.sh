#!/bin/sh

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
  echo '\nInstall tensorflow'
  python3 -m pip install tensorflow==2.1.0
  echo '\nInstall Python packages'
  python3 -m pip install -r requirements.txt
}

install_oasis() {
  echo '\n\nInstall OASIS'
  echo '\nInstall Gurobi'
  conda config --add channels http://conda.anaconda.org/gurobi
  conda install gurobi -y
  echo '\nInstall Mosek'
  conda install -c mosek mosek -y
  git clone https://github.com/j-friedrich/OASIS.git oasis
  cd oasis
  python3 setup.py build_ext --inplace
  python3 -m pip install -e .
  echo '\nInstall OASIS completed'
}

check_requirements
install_python_packages
install_oasis

echo '\nSetup completed.'
