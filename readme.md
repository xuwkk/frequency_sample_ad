# Efficient Sampling Method for Data-Driven Frequency Stability Constraint via Automatic Differentiation

This repository contains the code for the paper *"Efficient Sampling Method for Data-Driven Frequency Stability Constraint via Automatic Differentiation"* by Wangkun Xu, Qian Chen, Pudong Ge, Zhongda Chu, and Fei Teng. The authors are from control and power research group at Imperial College London.

## Requirements

Latest version of
```
torch, numpy, hydra-core
```

## Usage

Some utility tests are be found under the folder `test/`.

We use [hydra](https://hydra.cc/) to manage the configuration files. Therefore, you can change the default settings in `conf/configs.yaml` or by directly overwriting the settings in the command line.

To run the main algorithm, do
```
python main.py
```

To watch on specific sample (for example the first sample), do
```
python main.py ++watched_idx=0
```

such as
```
python test/test_gradient_fmad.py ++hyperparams.max_iter=1 ++hyperparams.batch_size=1
```
which will run this test file with only one iteration and one batch size.

## Comments

The power system dynamics are written in `torch.nn.Module` class to the future extension with neural network. It is possible to use pure `numpy` array to represent the power system dynamics and the AD can be implemented with `autograd` package.

