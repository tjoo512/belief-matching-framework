# Being Bayesian about Categorical Probability

This repository is the official implementation of ICML'2020 paper ["Being Bayesian about Categorical Probability."](https://arxiv.org/abs/2002.07965) 

The proposed framework, called belief matching framework, regards the categorical probability as a random variable and then constructs the Dirichlet target distribution over the categorical distribution by means of the Bayesian inference. Then, the neural network is trained to match its approximate distribution to the target distribution, which can be implemented by replacing only the softmax-cross entropy loss with the belief matching loss.

The code is designed to run on ```Python >= 3.5``` using the dependencies listed in ```requirements.txt```. You can install the dependencies by 

```
$ pip3 install -r requirements.txt.
```



## Training 

Experimental results presented in the paper can be reproduced by following instructions. 

**CIFAR**

Following scripts train ResNet-18 and ResNet-50 with the belief matching loss on CIFAR-10 and CIFAR-100 (use ```--coeff -1.0``` to train neural nets with the softmax-cross entropy loss).
```
$ python cifar_trainer.py --arch resnet18 --coeff 0.01 --dataset cifar10 --save-dir benchmark --gpu 0
$ python cifar_trainer.py --arch resnet18 --coeff 0.003 --dataset cifar100 --save-dir benchmark --gpu 0
$ python cifar_trainer.py --arch resnet50 --coeff 0.003 --dataset cifar10 --save-dir benchmark --gpu 0
$ python cifar_trainer.py --arch resnet50 --coeff 0.001 --dataset cifar100 --save-dir benchmark --gpu 0
```
**ImageNet** 

Following scripts train ResNext-50 and ResNext-101 with the belief matching loss on ImageNet (use ```--coeff -1.0``` to train neural nets with the softmax-cross entropy loss).
```
$ python imagenet_trainer.py --arch ResNext50 --coeff 0.001 --data DATA_DIR --save-dir benchmark
$ python imagenet_trainer.py --arch ResNext101 --coeff 0.0001 --data DATA_DIR --save-dir benchmark
```



Instructions and codes for transfer learning and semi-supervised learning are in ```transfer_learning``` and ```semi_supervised_learning```, respectively.



## Reference

Our code is based on the following public repositories:
* CIFAR: https://github.com/facebookresearch/mixup-cifar10
* ImageNet: https://github.com/hongyi-zhang/Fixup/tree/master/imagenet

