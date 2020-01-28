## Calcium Imaging GANs

### Requirements
It is recommended to install the codebase in a virtual environment ([virtualenv](https://pypi.org/project/virtualenv/) or [conda](https://conda.io/en/latest/)).

### Quick install
- Install all dependencies and packages with `setup.sh` script, works on both Linus and macOS.
```bash
sh setup.sh
```

### Manual setup
Install the following packages on your system:
- [TensorFlow](https://tensorflow.org)
- [j-friedrich/OASIS](https://github.com/j-friedrich/OASIS)
- [Gurobi](https://www.gurobi.com/)
- [MOSEK](https://www.mosek.com/)


### Dataset
- Place all raw calcium imaging data under `dataset/raw_data`
- Apply OASIS to infer spike train using [spike_train_inference.py](dataset/spike_train_inference.py)
```bash
python dataset/spike_train_inference.py --input_dir dataset/raw_data
```
- Generate `TFRecords` from a specific pickle file in `dataset/raw_data` with [generate_tfrecords.py](dataset/generate_tfrecords.py), normalize the data, preform segmentation and store the `TFrecords` in `output_dir`
```bash
python dataset/generate_tfrecords.py --input dataset/raw_data/ST260_Day4_signals4Bryan.pkl --output_dir dataset/sl128_normalize --normalize --sequence_length 256
```

### Train generative models
- Use `--help` flag to check all available hyperparameters
```bash
python main.py --input_dir dataset/sl256_normalize --output_dir runs/001 --epochs 100 --batch_size 64 --generator mlp --discriminator mlp --algoirthm wgan-gp --noise_dim 128 --plot_weights
```

### Visualize model
- Run `tensorboard` on `output_dir`
```bash
tensorboard --logdir runs/001
```