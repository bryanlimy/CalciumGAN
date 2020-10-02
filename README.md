## CalciumGAN: A Generative Adversarial Network Model for Synthesising Realistic Calcium Imaging Data of Neuronal Populations

### Citing this work
```
@misc{li2020calciumgan,
    title={CalciumGAN: A Generative Adversarial Network Model for Synthesising Realistic Calcium Imaging Data of Neuronal Populations},
    author={Bryan M. Li and Theoklitos Amvrosiadis and Nathalie Rochefort and Arno Onken},
    year={2020},
    eprint={2009.02707},
    archivePrefix={arXiv},
    primaryClass={q-bio.NC}
}
```

### Table of content
- [1. Installation](#1-installation)
- [2. Dataset](#2-dataset)
- [3. Training model](#3-train-model)
- [4. Spike analysis](#4-spike-analysis)
- [5. Visualization and Profiling](#5-visualization-and-profiling)
- [6. Hyper-parameter Search](#6-hyper-parameters-search)
    
---

### 1. Installation

### 1.1 Requirements
- It is recommended to install the codebase in a virtual environment, 
such as [conda](https://conda.io/en/latest/).

### 1.2 Quick install
- Create a new `conda` environment in Python 3.6
```bash
conda create -n calciumgan python=3.6
```
- Activate `calciumgan` virtual environment
```bash
conda activate calciumgan
```
- Install all dependencies and packages with `setup.sh` script, works on both Linus and macOS.
```bash
sh setup.sh
```

### 1.3 Manual setup
Install the following packages:
- [TensorFlow](https://tensorflow.org)
- [j-friedrich/OASIS](https://github.com/j-friedrich/OASIS)
- [Neo](https://github.com/NeuralEnsemble/python-neo)
- [Elephant](https://github.com/NeuralEnsemble/elephant)
- packages in `requirements.txt`
- code from [dg_python](https://github.com/mackelab/dg_python) are also being 
used for the dichotomized Gaussian model

---

### 2. Dataset
- Navigate to `dataset`
```bash
cd dataset
```

#### 2.1 Recorded Calcium Imaging Data
- Place all raw calcium imaging data under `dataset/raw_data`
- Apply OASIS to infer spike train
```bash
python spike_train_inference.py --input_dir raw_data
```
- Generate `TFRecords` from a specific pickle file `--input`, normalize the 
data, preform segmentation and store the `TFrecords` in `output_dir`. 
Use `--help` to see all available arguments.
```bash
python generate_tfrecords.py --input raw_data/signals.pkl --output_dir tfrecords/sl2048 --sequence_length 2048 --normalize
```

#### 2.2 Dichotomized Gaussian Artificial Data
- Generate artificial spike trains and calcium-like signals from the 
Dichomotized Gaussian distribution with the mean and covariance of data in 
`--input`, save the the output pickle file to `--output`. `TFRecords` in `--output_dir`. Use `--help` to 
see all available arguments.
```bash
python generate_dg_data.py --input raw_data/signals.pkl --output dg.pkl
```
- Generate `TFRecords` from a specific pickle file `--input`, normalize the 
data, preform segmentation and store the `TFrecords` in `output_dir`. 
Use `--help` to see all available arguments.
```bash
python generate_tfrecords.py --input dg.pkl --output_dir tfrecords/sl2048_dg --sequence_length 2048 --normalize
```

---

### 3. Train model
- To train CalciumGAN on the recorded calcium imaging data with the default 
hyper-parameters for 400 epochs. Checkpoints, generated data, model training
information are stored in `--output_dir`.
```bash
python main.py --input_dir dataset/tfrecords/sl2048 --output_dir runs/001 --epochs 400 --batch_size 128 --model calciumgan --algoirthm wgan-gp --noise_dim 32 --num_units 64 --kernel_size 24 --strides 2 --phase_shift 10 --layer_norm --mixed_precision --save_generated last 
```
- Use `--help` to check all available arguments. Mixed precision compute, 
TensorBoard profiling, hyper-parameter search are some of the features built 
into this codebase.
- The training command applies to both recorded data and dischotomized 
Gaussian artificial data.

---

### 4. Spike analysis

#### 4.1 Recorded Calcium Imaging Data
- Deconvolve the calcium signals to spike trains from generated data in 
`--output_dir`, then compute various spikes statistics. 
Use `--help` to check all available arguments.
```bash
python compute_metrics.py --output_dir runs/001
```
- All the plots can be found in `runs/001/metrics/plots`

#### 4.2 Dichotomized Gaussian Artificial Data
- Deconvolve the calcium signals to spike trains from generated data in 
`--output_dir`, then compute various spikes statistics. 
Use `--help` to check all available arguments.
```bash
python compute_dg_metrics.py --output_dir runs/002
```
- PLots of mean and covariance can be found in `diagrams/`

---

### 5. Visualization and Profiling
- Run `tensorboard`
```bash
tensorboard --logdir runs/001
```
- We have implemented profiling with 
[TensorFlow Profiler](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras) support.
You can enable profiling with `--profile` flag when training the model with `main.py`.

---

### 6. Hyper-parameters Search
- We have incorporated the [Hyperparameter Turning with Keras](https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams) feature. 
Modify the hyper-parameters you would like to test in `seasrch.py` and run
```bash
python search.py --input_dir dataset/tfrecords/sl2048 --output_dir runs/hparams_search --epochs 400 --mixed_precision
```
