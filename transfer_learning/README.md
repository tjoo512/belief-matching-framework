## Prepare Datasets

The following scripts download Cars dataset:

```
$ wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
$ wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
$ wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
$ tar -xvf cars_train.tgz
$ tar -xvf cars_test.tgz
$ tar -xvf car_devkit.tgz
$ cd devkit
$ wget http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat
```

The following scripts download Food-101 and preprocess the dataset:

```
$ wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
$ tar -zxvf food-101.tar.gz
$ python3 preprocessing_food.py
```



## Training

Following scripts load pre-trained ResNet-50 and train the last linear layer with the belief matching loss on CIFAR-10, Cars, and Food-101 (use ```--coeff -1.0``` to train neural nets with the softmax-cross entropy loss).

```
$ python main.py --dataset cifar10 --coeff 1e-2 --prior 1.0 --save_path transfer_cifar10_bm.pkl --gpu 0
$ python main.py --dataset scar --coeff 1e-2 --prior 1.0 --save_path transfer_cifar10_bm.pkl --gpu 0
$ python main.py --dataset food101 --coeff 1e-2 --prior 1.0 --save_path transfer_cifar10_bm.pkl --gpu 0

```

