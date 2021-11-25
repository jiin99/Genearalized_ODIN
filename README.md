# GeneralizedODIN

This is a implementation of generalized ODIN in [this paper](https://arxiv.org/pdf/2002.11297.pdf)
No official code has been released.

### Environment

* Python >= 3.6

* Pytorch >= 1.4

* CUDA >= 10.1

### Dataset
- In-dist :  CIFAR10/CIFAR100
- Out-dist : SVHN, LSUN, TinyImageNet

### Train a model 
```python
python main.py
```

### Test a model
```python
python test.py --network dense --save_root save_root --gpu 0
```
