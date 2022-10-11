# Layer-wise Orthogonal Training (LOT)

This repo provides the implementation of the paper "LOT: Layer-wise Orthogonal Training on Improving $\ell_2$ Certified Robustness". The code is improved based on the [code repo of SOC](https://github.com/singlasahil14/SOC).

## Prerequisites

Follow the instructions in [SOC](https://github.com/singlasahil14/SOC) to set up the environment.

## Training an LOT network

```python train_robust.py --conv-layer lot --activation ACT --block-size BLOCKS --dataset DATASET --gamma GAMMA --opt-level O0 --residual```
+ ACT: maxmin or hh1.
+ BLOCKS: 1, 2, 3, 4, 5, 6, 7, 8
+ DATASET: cifar10/cifar100.
+ GAMMA: certificate regularization coefficient
+ The LOT does not support O2 optimization as it requires a high float number precision.
+ Use ```--lln``` to enable last layer normalization

