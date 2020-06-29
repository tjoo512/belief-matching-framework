## Prepare Experiments 

Following scripts are required for running semi-supervised learning experiments.

```
python3 build_tfrecords.py --dataset_name=cifar10
python3 build_label_map.py --dataset_name=cifar10
```



## Training 

Following scripts train wide ResNet 28-2 on CIFAR-10 with the VAT and the pi-model on unlabeled examples and the softmax-cross entropy loss on labeled examples. 

```
$ mkdir experiments
$ mkdir experiment-logs
$ CUDA_VISIBLE_DEVICES=0 python3 train_model.py --verbosity=0 --primary_dataset_name='cifar10' --secondary_dataset_name='cifar10' --root_dir=./experiments/table-1-cifar10-4000-vat --n_labeled=4000 --consistency_model=vat --hparam_string=""  2>&1 | tee ./experiment-logs/table-1-cifar10-4000-vat_train.log
$ CUDA_VISIBLE_DEVICES=0 python3 train_model.py --verbosity=0 --primary_dataset_name='cifar10' --secondary_dataset_name='cifar10' --root_dir=./experiments/table-1-cifar10-4000-pi-model --n_labeled=4000 --consistency_model=pi_model --hparam_string=""  2>&1 | tee ./experiment-logs/table-1-cifar10-4000-pi-model_train.log
```

Following scripts train wide ResNet 28-2 on CIFAR-10 with distribution matching version of the VAT and the pi-model on unlabeled examples and the belief matching loss on labeled examples. 

```
$ CUDA_VISIBLE_DEVICES=0 python3 train_model.py --verbosity=0 --primary_dataset_name='cifar10' --secondary_dataset_name='cifar10' --root_dir=./experiments/table-1-cifar10-4000-vat_bm --n_labeled=4000 --consistency_model=vat_bm --hparam_string=""  2>&1 | tee ./experiment-logs/table-1-cifar10-4000-vat_train_bm.log
$ CUDA_VISIBLE_DEVICES=0 python3 train_model.py --verbosity=0 --primary_dataset_name='cifar10' --secondary_dataset_name='cifar10' --root_dir=./experiments/table-1-cifar10-4000-pi-model_bm --n_labeled=4000 --consistency_model=pi_model_bm --hparam_string=""  2>&1 | tee ./experiment-logs/table-1-cifar10-4000-pi-model_train_bm.log
```



## Load Results

Following scripts evaluate trained models and produce test accuracies.

```
$ CUDA_VISIBLE_DEVICES=0 python3 evaluate_model.py --split=test --verbosity=0 --primary_dataset_name='cifar10' --root_dir=./experiments/table-1-cifar10-4000-pi-model --consistency_model=pi_model --hparam_string=""  2>&1 | tee ./experiment-logs/table-1-cifar10-4000-pi-model_eval_test.log
$ CUDA_VISIBLE_DEVICES=0 python3 evaluate_model.py --split=test --verbosity=0 --primary_dataset_name='cifar10' --root_dir=./experiments/table-1-cifar10-4000-pi-model_bm --consistency_model=pi_model_bm --hparam_string=""  2>&1 | tee ./experiment-logs/table-1-cifar10-4000-pi-model_eval_test_bm.log
$ CUDA_VISIBLE_DEVICES=0 python3 evaluate_model.py --split=test --verbosity=0 --primary_dataset_name='cifar10' --root_dir=./experiments/table-1-cifar10-4000-vat --consistency_model=vat --hparam_string=""  2>&1 | tee ./experiment-logs/table-1-cifar10-4000-vat_eval_test.log
$ CUDA_VISIBLE_DEVICES=0 python3 evaluate_model.py --split=test --verbosity=0 --primary_dataset_name='cifar10' --root_dir=./experiments/table-1-cifar10-4000-vat_bm --consistency_model=vat_bm --hparam_string=""  2>&1 | tee ./experiment-logs/table-1-cifar10-4000-vat_eval_test_bm.log
```



## Reference

Semi-supervised learning experiments are based on the following public repository: https://github.com/brain-research/realistic-ssl-evaluation

