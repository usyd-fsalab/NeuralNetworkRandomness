# NeuralNetworkRandomness

This is the code repo for [Randomness in Neural Network Training: Characterizing the Impact of Tooling](https://todo). 

# Initialization

```
git clone --recursive https://github.com/usyd-fsalab/NeuralNetworkRandomness.git
cd NeuralNetworkRandomness
pip install -r requirements.txt
```

# Run Experiment
## Three layer small CNN
```
# ALGO+IMPL
python -m training_script.smallcnn --ckpt_folder ./logs_algoimpl/ --lr 4e-4 --batch_size 128 --eppchs 200 
# ALGO
python -m training_script.smallcnn --ckpt_folder ./logs_algo/ --lr 4e-4 --batch_size 128 --eppchs 200 --deterministic_tf
# IMPL
python -m training_script.smallcnn --ckpt_folder ./logs_impl/ --lr 4e-4 --batch_size 128 --eppchs 200 --deterministic_init --deterministic_input
# Control (fully deterministic)
python -m training_script.smallcnn --ckpt_folder ./logs_control/ --lr 4e-4 --batch_size 128 --eppchs 200 --deterministic_init --deterministic_input --deterministic_tf
```

## ResNet18 CIFAR10
```
# ALGO+IMPL
python -m training_script.resnet18 --ckpt_folder ./logs_algoimpl/ --lr 4e-4 --batch_size 128 --num_epoch 200 
# ALGO
python -m training_script.resnet18 --ckpt_folder ./logs_algo/ --lr 4e-4 --batch_size 128 --num_epoch 200 --deterministic_tf
# IMPL
python -m training_script.resnet18 --ckpt_folder ./logs_impl/ --lr 4e-4 --batch_size 128 --num_epoch 200 --deterministic_init --deterministic_input
# Control (fully deterministic)
python -m training_script.resnet18 --ckpt_folder ./logs_control/ --lr 4e-4 --batch_size 128 --num_epoch 200 --deterministic_init --deterministic_input --deterministic_tf
```

## ResNet18 CIFAR100
```
# ALGO+IMPL
python -m training_script.resnet18 --dataset cifar100 --ckpt_folder ./logs_algoimpl/ --lr 4e-4 --batch_size 128 --num_epoch 200 
# ALGO
python -m training_script.resnet18 --dataset cifar100 --ckpt_folder ./logs_algo/ --lr 4e-4 --batch_size 128 --num_epoch 200 --deterministic_tf
# IMPL
python -m training_script.resnet18 --dataset cifar100 --ckpt_folder ./logs_impl/ --lr 4e-4 --batch_size 128 --num_epoch 200 --deterministic_init --deterministic_input
# Control (fully deterministic)
python -m training_script.resnet18 --dataset cifar100 --ckpt_folder ./logs_control/ --lr 4e-4 --batch_size 128 --num_epoch 200 --deterministic_init --deterministic_input --deterministic_tf
```

## ResNet50 ImageNet
1. Config how many GPUs is available for trianing in ./models/official/vision/image_classification/configs/examples/resnet/imagenet/gpu.yaml (default is 4)
2. Config other parameter in gpu.ymal file if needed. Do not change it if you would like to use same configuration in paper.
```
cd models
export PYTHONPATH=/absolute/path/to/NeuralNetworkRandomness/models:${PYTHONPATH}
# ImageNet dataset is expected stored in format that can be parsed by tensorflow_dataset (tfds.load)
export IMAGENET_DATADIR=/path/to/data 
# ALGO+IMPL
python ./official/vision/image_classification/classifier_trainer.py --mode=train_and_eval --model_type=resnet --dataset=imagenet --model_dir=./logs_algo_impl/ --data_dir=$IMAGENET_DATADIR --config_file=./official/vision/image_classification/configs/examples/resnet/imagenet/gpu.yaml
# ALGO
python ./official/vision/image_classification/classifier_trainer.py --mode=train_and_eval --model_type=resnet --dataset=imagenet --model_dir=./logs_algo_impl/ --data_dir=$IMAGENET_DATADIR --config_file=./official/vision/image_classification/configs/examples/resnet/imagenet/gpu.yaml --deterministic_tf
# IMPL
python ./official/vision/image_classification/classifier_trainer.py --mode=train_and_eval --model_type=resnet --dataset=imagenet --model_dir=./logs_algo_impl/ --data_dir=$IMAGENET_DATADIR --config_file=./official/vision/image_classification/configs/examples/resnet/imagenet/gpu.yaml --deterministic_init --deterministic_input
# Control
python ./official/vision/image_classification/classifier_trainer.py --mode=train_and_eval --model_type=resnet --dataset=imagenet --model_dir=./logs_algo_impl/ --data_dir=$IMAGENET_DATADIR --config_file=./official/vision/image_classification/configs/examples/resnet/imagenet/gpu.yaml --deterministic_init --deterministic_input --deterministic_tf
```

## Deterministic Overhead
We use nvprof profiler to collect GPU time duration. The performance data is collected over 100 training iterations.
```
# profiling model in non-deterministic training
nvprof python -m training_script --model_name resnet50 --batch_size 128 --data_dir /path/to/imagenet/data/in/tfds/format

# profiling model in deterministic training
nvprof python -m training_script --model_name resnet50 --batch_size 128 --data_dir /path/to/imagenet/data/in/tfds/format --deterministic_tf
```

# Cite This Paper

```
```