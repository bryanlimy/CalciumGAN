#!/bin/sh

current_dir="$(pwd)"
macOS=false

check_requirements() {
  case "$(uname -s)" in
    Darwin)
      echo 'Installing on macOS.'
      export CFLAGS='-stdlib=libc++'
      macOS=true
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
  echo 'Install tensorflow'
  if [ "$macOS" = "true" ]; then
    python3 -m pip install tensorflow==2.2.0
  else
    conda install -c anaconda tensorflow-gpu==2.2.0 -y
  fi
  echo 'Install Python packages'
  python3 -m pip install -r requirements.txt
}

install_oasis() {
  echo 'Install OASIS'
  cd "$current_dir" || exit 1
  echo 'Install Gurobi'
  conda config --add channels http://conda.anaconda.org/gurobi
  conda install gurobi -y
  echo 'Install Mosek'
  conda install -c mosek mosek -y
  git clone https://github.com/j-friedrich/OASIS.git oasis
  cd oasis || exit 1
  python3 setup.py build_ext --inplace
  python3 -m pip install -e .
  echo 'Install OASIS completed'
}

install_elephant() {
  echo 'Install Elephant'
  cd "$current_dir" || exit 1
  git clone git://github.com/NeuralEnsemble/elephant.git elephant
  cd elephant || exit 1
  python3 setup.py install
  echo 'Install Elephant completed'
}

check_requirements
install_python_packages
install_oasis
install_elephant

echo 'Setup completed.'
