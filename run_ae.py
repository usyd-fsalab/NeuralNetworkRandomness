import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--gpus', nargs='+', type=str, required=True)
parser.add_argument('--experiments', nargs='+', required=True, choices=['Figure1', 'Figure2', 'Figure3', 'Figure4', 'Figure5', 'Figure6', 'Figure8'])
parser.add_argument('--fast', action='store_true')
parser.add_argument('--imagenet_data_dir', type=str, default=None)

args = parser.parse_args()

from pyexpat import model
import shlex, subprocess
import os
import csv
from enum import Enum
from enum import IntEnum
import subprocess
import glob
import numpy as np
import tensorflow as tf

class DeterministicSetting(Enum):
    Default = 'default'
    ALGO_Noise_Only = 'algo'
    IMPL_Noise_Only = 'impl'
    Control = 'control'

class Models(Enum):
    SmallCNN = 'smallcnn'
    ResNet18 = 'resnet18'
    ResNet50 = 'resnet50'

class Datasets(Enum):
    CIFAR10 = 'cifar10'
    CIFAR100 = 'cifar100'
    CelebA = 'celeba'
    ImageNet = 'imagenet'

class ModelSet(IntEnum):
    SmallCNN_CIFAR10 = 1 << 0
    SmallCNN_CIFAR100 = 1 << 1
    ResNet18_CIFAR10 = 1 << 2
    ResNet18_CIFAR100 = 1 << 3
    ResNet50_ImageNet = 1 << 4
    SmallCNN_CIFAR10_BN = 1 << 5
    ResNet18_CelebA = 1 << 6

class ModelConfig:
    def __init__(self, _model : Models, _dataset : Datasets, _determinism : DeterministicSetting, _num_models : int, _additional_args : list = None) -> None:
        self.model = _model
        self.dataset = _dataset
        self.determinism = _determinism
        self.num_models = _num_models
        self.additional_args = _additional_args
        
    def __eq__(self, __o: object) -> bool:
        return self.model == __o.model and \
                self.dataset == __o.dataset and  \
                self.determinism == __o.determinism and \
                self.num_models == __o.num_models and \
                self.additional_args == __o.additional_args
            
    def to_string(self) -> str:
        additional_args_str = ''
        if self.additional_args != None and len(self.additional_args) > 0:
            additional_args_str = ' '.join(self.additional_args)

        return 'Model: {}, Dataset: {}, Determinism: {}, num_models: {}, additional_args:{}'.format(self.model.value, self.dataset.value, self.determinism.value, self.num_models.value, additional_args_str)

    def to_command_lines(self) -> list:
        commands = []
        if self.dataset == Datasets.CIFAR10:
            command_line_template = 'python -m src.training_script.{} --ckpt_folder ./{}{}/ --lr 4e-4 --batch_size 128 --epochs 200'.format(self.model.value, self.get_ckpt_dir_prefix(), r'{}')
        if self.dataset == Datasets.CIFAR100:
            command_line_template = 'python -m src.training_script.{} --ckpt_folder ./{}{}/ --dataset cifar100 --lr 4e-4 --batch_size 128 --epochs 200'.format(self.model.value, self.get_ckpt_dir_prefix(), r'{}')
        if self.dataset == Datasets.CelebA:
            command_line_template = 'python -m src.training_script.resnet_celeba --ckpt_folder ./{}{}/'.format(self.get_ckpt_dir_prefix(), r'{}')
        if self.dataset == Datasets.ImageNet:
            command_line_template = 'python ./official/vision/image_classification/classifier_trainer.py --mode=train_and_eval --model_type=resnet --dataset=imagenet --model_dir={}{}/ --data_dir={}'.format(self.get_ckpt_dir_prefix(), r'{}', args.imagenet_data_dir)

        if self.determinism == DeterministicSetting.ALGO_Noise_Only:
            command_line_template += ' --deterministic_tf'
        if self.determinism == DeterministicSetting.IMPL_Noise_Only:
            command_line_template += ' --deterministic_init --deterministic_input'
        if self.determinism == DeterministicSetting.Control:
            command_line_template += ' --deterministic_init --deterministic_input --deterministic_tf'
        
        if self.additional_args != None:
            for additional_arg in self.additional_args:
                command_line_template += f' {additional_arg}'

        for idx in range(self.num_models):
            commands.append(command_line_template.format(idx))
        
        return commands
    def get_ckpt_dir_prefix(self) -> str:
        prefix = 'logs_{}_{}_{}_'.format(self.model.value, self.dataset.value, self.determinism.value)
        if self.additional_args != None and '--bn' in self.additional_args:
            prefix += '_bn'
        if self.dataset == Datasets.ImageNet:
            return os.path.join(os.getcwd(), prefix)
        return prefix

    def add_additional_arg(self, arg) -> None:
        if self.additional_args == None:
            self.additional_args = []
        self.additional_args.append(arg)

    model : Models
    dataset : Datasets
    determinism : DeterministicSetting
    num_models : int
    additional_args : list



def get_standard_deviation_of_accuracy(config : ModelConfig):
    model_dir = config.get_ckpt_dir_prefix()
    acc_list = []
    for id in range(config.num_models):
        with open(os.path.join(model_dir + str(id), 'log.csv')) as file:
            csv_iter = iter(csv.reader(file, delimiter=','))
            header = next(csv_iter)
            col_id = header.index('val_accuracy')
            acc = -1

            for row in csv_iter:
                acc = float(row[col_id])
            
            acc_list.append(acc)

    return np.std(acc_list)

def get_standard_deviation_of_fpr(config : ModelConfig):
    model_dir = config.get_ckpt_dir_prefix()
    fpr_list = []
    for id in range(config.num_models):
        with open(os.path.join(model_dir + str(id), 'log.csv')) as file:
            csv_iter = iter(csv.reader(file, delimiter=','))
            header = next(csv_iter)
            fp_colid = header.index('val_false_positives')
            tn_colid = header.index('val_true_negatives')
            fpr = -1

            for row in csv_iter:
                fpr = float(row[fp_colid]) / (float(row[fp_colid]) + float(row[tn_colid]))
            
            fpr_list.append(fpr)

    return np.std(fpr_list)

def get_standard_deviation_of_fnr(config : ModelConfig):
    model_dir = config.get_ckpt_dir_prefix()
    fnr_list = []
    for id in range(config.num_models):
        with open(os.path.join(model_dir + str(id), 'log.csv')) as file:
            csv_iter = iter(csv.reader(file, delimiter=','))
            header = next(csv_iter)
            fn_colid = header.index('val_false_negatives')
            fp_colid = header.index('val_false_positives')
            tn_colid = header.index('val_true_negatives')
            fnr = -1

            for row in csv_iter:
                fnr = float(row[fn_colid]) / (float(row[fp_colid]) + float(row[tn_colid]))
            
            fnr_list.append(fnr)

    return np.std(fnr_list)

def get_prediction_churn(config : ModelConfig):
    def get_churn(arr1, arr2):
        assert arr1.shape == arr2.shape, 'Shape mismatch'
        return np.sum(arr1 == arr2) / arr1.size
    
    model_dir = config.get_ckpt_dir_prefix()

    predictions = []

    for id in range(config.num_models):
        pred_files = glob.glob(os.path.join(model_dir + str(id), 'pred*.txt'))
        pred_files.sort()
        pred_file = pred_files[-1]
        predictions.append(np.loadtxt(pred_file))

    churn_tot = 0.0
    cnt = 0

    for i in range(config.num_models):
        for j in range(config.num_models):
            if j > i:
                cnt += 1
                churn_tot += get_churn(predictions[i], predictions[j])
    
    return churn_tot / cnt

def get_l2_norm(config : ModelConfig):
    model_dir = config.get_ckpt_dir_prefix()
    weights = []
    for id in range(config.num_models):
        model_files = glob.glob(os.path.join(model_dir + str(id), 'ckpt*.h5'))
        model_files.sort()
        model_file = model_files[-1]
        model = tf.keras.models.load_model(model_file, compile='h5')
        weight = []
        for w in model.weights:
            weight.extend(list(np.reshape(w, -1)))
        
        weights.append(weight)

    cnt = 0
    accum = 0.0

    for i in range(config.num_models):
        for j in range(config.num_models):
            if j > i:
                accum += np.linalg.norm(np.array(weights[i]) - np.array(weights[j]))
                cnt += 1

    return accum / cnt

def train_models() -> None:
    print('{s:{c}^{n}}'.format(s = 'Train models for specified experiments', c = '=', n = '100'))

    models_to_train = 0
    model_config_list = []
    imagenet_model_config_list = []

    if 'Figure1' in args.experiments:
        models_to_train = models_to_train | ModelSet.SmallCNN_CIFAR10
        models_to_train |= ModelSet.ResNet18_CIFAR10
        models_to_train |= ModelSet.ResNet18_CIFAR100
        models_to_train |= ModelSet.ResNet50_ImageNet
    if 'Figure2' in args.experiments:
        models_to_train |= ModelSet.SmallCNN_CIFAR10
        models_to_train |= ModelSet.SmallCNN_CIFAR10_BN
    if 'Figure3' in args.experiments:
        models_to_train |= ModelSet.SmallCNN_CIFAR10
        models_to_train |= ModelSet.SmallCNN_CIFAR100
        models_to_train |= ModelSet.ResNet18_CIFAR10
        models_to_train |= ModelSet.ResNet18_CIFAR100
    if 'Figure4' in args.experiments:
        models_to_train |= ModelSet.ResNet18_CelebA
    if 'Figure5' in args.experiments:
        models_to_train |= ModelSet.ResNet18_CIFAR100

    if models_to_train & ModelSet.SmallCNN_CIFAR10:
        model_config_list.append(ModelConfig(Models.SmallCNN, Datasets.CIFAR10, DeterministicSetting.Default, 10))
        model_config_list.append(ModelConfig(Models.SmallCNN, Datasets.CIFAR10, DeterministicSetting.ALGO_Noise_Only, 10))
        model_config_list.append(ModelConfig(Models.SmallCNN, Datasets.CIFAR10, DeterministicSetting.IMPL_Noise_Only, 10))
    if models_to_train & ModelSet.SmallCNN_CIFAR100:
        model_config_list.append(ModelConfig(Models.SmallCNN, Datasets.CIFAR100, DeterministicSetting.Default, 10))
        model_config_list.append(ModelConfig(Models.SmallCNN, Datasets.CIFAR100, DeterministicSetting.ALGO_Noise_Only, 10))
        model_config_list.append(ModelConfig(Models.SmallCNN, Datasets.CIFAR100, DeterministicSetting.IMPL_Noise_Only, 10))
    if models_to_train & ModelSet.SmallCNN_CIFAR10_BN:
        model_config_list.append(ModelConfig(Models.SmallCNN, Datasets.CIFAR10, DeterministicSetting.Default, 10, ['--bn']))
        model_config_list.append(ModelConfig(Models.SmallCNN, Datasets.CIFAR10, DeterministicSetting.ALGO_Noise_Only, 10, ['--bn']))
        model_config_list.append(ModelConfig(Models.SmallCNN, Datasets.CIFAR10, DeterministicSetting.IMPL_Noise_Only, 10, ['--bn']))
    if models_to_train & ModelSet.ResNet18_CIFAR10:
        model_config_list.append(ModelConfig(Models.ResNet18, Datasets.CIFAR10, DeterministicSetting.Default, 10))
        model_config_list.append(ModelConfig(Models.ResNet18, Datasets.CIFAR10, DeterministicSetting.ALGO_Noise_Only, 10))
        model_config_list.append(ModelConfig(Models.ResNet18, Datasets.CIFAR10, DeterministicSetting.IMPL_Noise_Only, 10))
    if models_to_train & ModelSet.ResNet18_CIFAR100:
        model_config_list.append(ModelConfig(Models.ResNet18, Datasets.CIFAR100, DeterministicSetting.Default, 10))
        model_config_list.append(ModelConfig(Models.ResNet18, Datasets.CIFAR100, DeterministicSetting.ALGO_Noise_Only, 10))
        model_config_list.append(ModelConfig(Models.ResNet18, Datasets.CIFAR100, DeterministicSetting.IMPL_Noise_Only, 10))
    if models_to_train & ModelSet.ResNet50_ImageNet:
        imagenet_model_config_list.append(ModelConfig(Models.ResNet50, Datasets.ImageNet, DeterministicSetting.Default, 5))
        imagenet_model_config_list.append(ModelConfig(Models.ResNet50, Datasets.ImageNet, DeterministicSetting.ALGO_Noise_Only, 5))
        imagenet_model_config_list.append(ModelConfig(Models.ResNet50, Datasets.ImageNet, DeterministicSetting.IMPL_Noise_Only, 5))
    if models_to_train & ModelSet.ResNet18_CelebA:
        model_config_list.append(ModelConfig(Models.ResNet18, Datasets.CelebA, DeterministicSetting.Default, 10))
        model_config_list.append(ModelConfig(Models.ResNet18, Datasets.CelebA, DeterministicSetting.ALGO_Noise_Only, 10))
        model_config_list.append(ModelConfig(Models.ResNet18, Datasets.CelebA, DeterministicSetting.IMPL_Noise_Only, 10))

    if args.fast:
        for config in model_config_list:
            config.add_additional_arg('--fast')
        for config in imagenet_model_config_list:
            config.add_additional_arg('--config_file=./official/vision/image_classification/configs/examples/resnet/imagenet/gpu_ae.yaml')
    else:
        for config in imagenet_model_config_list:
            config.add_additional_arg('--config_file=./official/vision/image_classification/configs/examples/resnet/imagenet/gpu.yaml')

    train_command_lines = []
    for config in model_config_list:
        for line in config.to_command_lines():
            train_command_lines.append(line)
    
    command_line_iter = iter(train_command_lines)
    n_gpu = len(args.gpus)
    my_env = os.environ.copy()

    while True:
        process_list = []
        for i in range(n_gpu):
            gpu = args.gpus[i]
            command_line = next(command_line_iter, None)
            if command_line == None:
                break

            print(f'Execute: {command_line}')
            
            train_args = shlex.split(command_line)
            my_env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            my_env['TF_CPP_MIN_LOG_LEVEL'] = '3'
            # p = subprocess.Popen(train_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=my_env)
            p = subprocess.Popen(train_args, env=my_env)
            process_list.append(p)

        for p in process_list:
            p.wait()

        if len(process_list) < n_gpu:
            break

    if len(imagenet_model_config_list) > 0:
        assert args.imagenet_data_dir != None, 'ImageNet data dir should be provided'

        working_dir = os.path.join(os.getcwd(), 'src/models')

        my_env['TF_CPP_MIN_LOG_LEVEL'] = '3'
        my_env['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpus)
        my_env['PYTHONPATH'] = f'{working_dir}'
        my_env['IMAGENET_DATADIR'] = args.imagenet_data_dir
        
        for config in imagenet_model_config_list:
            for command_line in config.to_command_lines():
                print(f'Execute: {command_line}')
                train_args = shlex.split(command_line)
                p = subprocess.Popen(train_args, env=my_env, cwd=working_dir)
                p.wait()
    
    print('{s:{c}^{n}}'.format(s = 'Done', c = '=', n = '100'))

def figure1() -> None:
    print('{s:{c}^{n}}'.format(s = 'Experiment Figure1', c = '=', n = '100'))
    smallcnn_cifar10_default = ModelConfig(Models.SmallCNN, Datasets.CIFAR10, DeterministicSetting.Default, 10)
    smallcnn_cifar10_algo = ModelConfig(Models.SmallCNN, Datasets.CIFAR10, DeterministicSetting.ALGO_Noise_Only, 10)
    smallcnn_cifar10_impl = ModelConfig(Models.SmallCNN, Datasets.CIFAR10, DeterministicSetting.IMPL_Noise_Only, 10)

    resnet18_cifar10_default = ModelConfig(Models.ResNet18, Datasets.CIFAR10, DeterministicSetting.Default, 10)
    resnet18_cifar10_algo = ModelConfig(Models.ResNet18, Datasets.CIFAR10, DeterministicSetting.ALGO_Noise_Only, 10)
    resnet18_cifar10_impl = ModelConfig(Models.ResNet18, Datasets.CIFAR10, DeterministicSetting.IMPL_Noise_Only, 10)

    resnet18_cifar100_default = ModelConfig(Models.ResNet18, Datasets.CIFAR100, DeterministicSetting.Default, 10)
    resnet18_cifar100_algo = ModelConfig(Models.ResNet18, Datasets.CIFAR100, DeterministicSetting.ALGO_Noise_Only, 10)
    resnet18_cifar100_impl = ModelConfig(Models.ResNet18, Datasets.CIFAR100, DeterministicSetting.IMPL_Noise_Only, 10)

    resnet50_imagenet_default = ModelConfig(Models.ResNet50, Datasets.ImageNet, DeterministicSetting.Default, 5)
    resnet50_imagenet_algo = ModelConfig(Models.ResNet50, Datasets.ImageNet, DeterministicSetting.ALGO_Noise_Only, 5)
    resnet50_imagenet_impl = ModelConfig(Models.ResNet50, Datasets.ImageNet, DeterministicSetting.IMPL_Noise_Only, 5)

    with open('./results/figure1.csv', 'w') as file:
        csvwriter = csv.writer(file) 
        csvwriter.writerow(['Network', 'Dataset', 'Metric', 'Setting', 'Value'])
        ################## small cnn cifar10
        csvwriter.writerow(['SmallCNN', 'CIFAR10', 'STDDEV(Accuracy)', 'Default', get_standard_deviation_of_accuracy(smallcnn_cifar10_default)])
        csvwriter.writerow(['SmallCNN', 'CIFAR10', 'STDDEV(Accuracy)', 'ALGO', get_standard_deviation_of_accuracy(smallcnn_cifar10_algo)])
        csvwriter.writerow(['SmallCNN', 'CIFAR10', 'STDDEV(Accuracy)', 'IMPL', get_standard_deviation_of_accuracy(smallcnn_cifar10_impl)])

        csvwriter.writerow(['SmallCNN', 'CIFAR10', 'Churn', 'Default', get_prediction_churn(smallcnn_cifar10_default)])
        csvwriter.writerow(['SmallCNN', 'CIFAR10', 'Churn', 'ALGO', get_prediction_churn(smallcnn_cifar10_algo)])
        csvwriter.writerow(['SmallCNN', 'CIFAR10', 'Churn', 'IMPL', get_prediction_churn(smallcnn_cifar10_impl)])

        csvwriter.writerow(['SmallCNN', 'CIFAR10', 'L2 Norm', 'Default', get_l2_norm(smallcnn_cifar10_default)])
        csvwriter.writerow(['SmallCNN', 'CIFAR10', 'L2 Norm', 'ALGO', get_l2_norm(smallcnn_cifar10_algo)])
        csvwriter.writerow(['SmallCNN', 'CIFAR10', 'L2 Norm', 'IMPL', get_l2_norm(smallcnn_cifar10_impl)])

        ################## resnet18 cifar10
        csvwriter.writerow(['ResNet18', 'CIFAR10', 'STDDEV(Accuracy)', 'Default', get_standard_deviation_of_accuracy(resnet18_cifar10_default)])
        csvwriter.writerow(['ResNet18', 'CIFAR10', 'STDDEV(Accuracy)', 'ALGO', get_standard_deviation_of_accuracy(resnet18_cifar10_algo)])
        csvwriter.writerow(['ResNet18', 'CIFAR10', 'STDDEV(Accuracy)', 'IMPL', get_standard_deviation_of_accuracy(resnet18_cifar10_impl)])

        csvwriter.writerow(['ResNet18', 'CIFAR10', 'Churn', 'Default', get_prediction_churn(resnet18_cifar10_default)])
        csvwriter.writerow(['ResNet18', 'CIFAR10', 'Churn', 'ALGO', get_prediction_churn(resnet18_cifar10_algo)])
        csvwriter.writerow(['ResNet18', 'CIFAR10', 'Churn', 'IMPL', get_prediction_churn(resnet18_cifar10_impl)])

        csvwriter.writerow(['ResNet18', 'CIFAR10', 'L2 Norm', 'Default', get_l2_norm(resnet18_cifar10_default)])
        csvwriter.writerow(['ResNet18', 'CIFAR10', 'L2 Norm', 'ALGO', get_l2_norm(resnet18_cifar10_algo)])
        csvwriter.writerow(['ResNet18', 'CIFAR10', 'L2 Norm', 'IMPL', get_l2_norm(resnet18_cifar10_impl)])

        ################## resnet18 cifar100
        csvwriter.writerow(['ResNet18', 'CIFAR100', 'STDDEV(Accuracy)', 'Default', get_standard_deviation_of_accuracy(resnet18_cifar100_default)])
        csvwriter.writerow(['ResNet18', 'CIFAR100', 'STDDEV(Accuracy)', 'ALGO', get_standard_deviation_of_accuracy(resnet18_cifar100_algo)])
        csvwriter.writerow(['ResNet18', 'CIFAR100', 'STDDEV(Accuracy)', 'IMPL', get_standard_deviation_of_accuracy(resnet18_cifar100_impl)])

        csvwriter.writerow(['ResNet18', 'CIFAR100', 'Churn', 'Default', get_prediction_churn(resnet18_cifar100_default)])
        csvwriter.writerow(['ResNet18', 'CIFAR100', 'Churn', 'ALGO', get_prediction_churn(resnet18_cifar100_algo)])
        csvwriter.writerow(['ResNet18', 'CIFAR100', 'Churn', 'IMPL', get_prediction_churn(resnet18_cifar100_impl)])

        csvwriter.writerow(['ResNet18', 'CIFAR100', 'L2 Norm', 'Default', get_l2_norm(resnet18_cifar100_default)])
        csvwriter.writerow(['ResNet18', 'CIFAR100', 'L2 Norm', 'ALGO', get_l2_norm(resnet18_cifar100_algo)])
        csvwriter.writerow(['ResNet18', 'CIFAR100', 'L2 Norm', 'IMPL', get_l2_norm(resnet18_cifar100_impl)])

        ################## resnet50 imagenet
        csvwriter.writerow(['ResNet18', 'ImageNet', 'STDDEV(Accuracy)', 'Default', get_standard_deviation_of_accuracy(resnet50_imagenet_default)])
        csvwriter.writerow(['ResNet18', 'ImageNet', 'STDDEV(Accuracy)', 'ALGO', get_standard_deviation_of_accuracy(resnet50_imagenet_algo)])
        csvwriter.writerow(['ResNet18', 'ImageNet', 'STDDEV(Accuracy)', 'IMPL', get_standard_deviation_of_accuracy(resnet50_imagenet_impl)])

        csvwriter.writerow(['ResNet18', 'ImageNet', 'Churn', 'Default', get_prediction_churn(resnet50_imagenet_default)])
        csvwriter.writerow(['ResNet18', 'ImageNet', 'Churn', 'ALGO', get_prediction_churn(resnet50_imagenet_algo)])
        csvwriter.writerow(['ResNet18', 'ImageNet', 'Churn', 'IMPL', get_prediction_churn(resnet50_imagenet_impl)])

        csvwriter.writerow(['ResNet18', 'ImageNet', 'L2 Norm', 'Default', get_l2_norm(resnet50_imagenet_default)])
        csvwriter.writerow(['ResNet18', 'ImageNet', 'L2 Norm', 'ALGO', get_l2_norm(resnet50_imagenet_algo)])
        csvwriter.writerow(['ResNet18', 'ImageNet', 'L2 Norm', 'IMPL', get_l2_norm(resnet50_imagenet_impl)])

    print('{s:{c}^{n}}'.format(s = 'Done', c = '=', n = '100'))

def figure2() -> None:
    print('{s:{c}^{n}}'.format(s = 'Experiment Figure2', c = '=', n = '100'))

    default_bn = ModelConfig(Models.SmallCNN, Datasets.CIFAR10, DeterministicSetting.Default, 10, ['--bn'])
    algo_bn = ModelConfig(Models.SmallCNN, Datasets.CIFAR10, DeterministicSetting.ALGO_Noise_Only, 10, ['--bn'])
    impl_bn = ModelConfig(Models.SmallCNN, Datasets.CIFAR10, DeterministicSetting.IMPL_Noise_Only, 10, ['--bn'])

    default = ModelConfig(Models.SmallCNN, Datasets.CIFAR10, DeterministicSetting.Default, 10)
    algo = ModelConfig(Models.SmallCNN, Datasets.CIFAR10, DeterministicSetting.ALGO_Noise_Only, 10)
    impl = ModelConfig(Models.SmallCNN, Datasets.CIFAR10, DeterministicSetting.IMPL_Noise_Only, 10)

    with open('./results/figure2.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Setting', 'BN', 'Value'])

        # stddev(accuracy)
        writer.writerow(['STDDEV(Accuracy)', 'Default', 'Yes', get_standard_deviation_of_accuracy(default_bn)])
        writer.writerow(['STDDEV(Accuracy)', 'Default', 'No', get_standard_deviation_of_accuracy(default)])

        writer.writerow(['STDDEV(Accuracy)', 'ALGO', 'Yes', get_standard_deviation_of_accuracy(algo_bn)])
        writer.writerow(['STDDEV(Accuracy)', 'ALGO', 'No', get_standard_deviation_of_accuracy(algo)])

        writer.writerow(['STDDEV(Accuracy)', 'IMPL', 'Yes', get_standard_deviation_of_accuracy(impl_bn)])
        writer.writerow(['STDDEV(Accuracy)', 'IMPL', 'No', get_standard_deviation_of_accuracy(impl)])

        # churn
        writer.writerow(['Churn', 'Default', 'Yes', get_prediction_churn(default_bn)])
        writer.writerow(['Churn', 'Default', 'No', get_prediction_churn(default)])

        writer.writerow(['Churn', 'ALGO', 'Yes', get_prediction_churn(algo_bn)])
        writer.writerow(['Churn', 'ALGO', 'No', get_prediction_churn(algo)])

        writer.writerow(['Churn', 'IMPL', 'Yes', get_prediction_churn(impl_bn)])
        writer.writerow(['Churn', 'IMPL', 'No', get_prediction_churn(impl)])

        # L2 Norm
        writer.writerow(['L2 Norm', 'Default', 'Yes', get_l2_norm(default_bn)])
        writer.writerow(['L2 Norm', 'Default', 'No', get_l2_norm(default)])

        writer.writerow(['L2 Norm', 'ALGO', 'Yes', get_l2_norm(algo_bn)])
        writer.writerow(['L2 Norm', 'ALGO', 'No', get_l2_norm(algo)])

        writer.writerow(['L2 Norm', 'IMPL', 'Yes', get_l2_norm(impl_bn)])
        writer.writerow(['L2 Norm', 'IMPL', 'No', get_l2_norm(impl)])

    print('{s:{c}^{n}}'.format(s = 'Done', c = '=', n = '100'))


def figure5() -> None:
    print('{s:{c}^{n}}'.format(s = 'Experiment Figure5', c = '=', n = '100'))
    resnet18_cifar_100_default = ModelConfig(Models.ResNet18, Datasets.CIFAR100, DeterministicSetting.Default, 10)
    resnet18_cifar_100_algo = ModelConfig(Models.ResNet18, Datasets.CIFAR100, DeterministicSetting.ALGO_Noise_Only, 10)
    resnet18_cifar_100_impl = ModelConfig(Models.ResNet18, Datasets.CIFAR100, DeterministicSetting.IMPL_Noise_Only, 10)

    with open('results/figure5.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Setting', 'Value'])
        writer.writerow(['STDDEV(Accuracy)', 'Default', get_standard_deviation_of_accuracy(resnet18_cifar_100_default)])
        writer.writerow(['STDDEV(Accuracy)', 'ALGO', get_standard_deviation_of_accuracy(resnet18_cifar_100_algo)])
        writer.writerow(['STDDEV(Accuracy)', 'IMPL', get_standard_deviation_of_accuracy(resnet18_cifar_100_impl)])

        writer.writerow(['Churn', 'Default', get_prediction_churn(resnet18_cifar_100_default)])
        writer.writerow(['Churn', 'ALGO', get_prediction_churn(resnet18_cifar_100_algo)])
        writer.writerow(['Churn', 'IMPL', get_prediction_churn(resnet18_cifar_100_impl)])

        writer.writerow(['L2 Norm', 'Default', get_l2_norm(resnet18_cifar_100_default)])
        writer.writerow(['L2 Norm', 'ALGO', get_l2_norm(resnet18_cifar_100_algo)])
        writer.writerow(['L2 Norm', 'IMPL', get_l2_norm(resnet18_cifar_100_impl)])

    print('{s:{c}^{n}}'.format(s = 'Done', c = '=', n = '100'))

def get_execution_time(text):
    lines = text.decode('utf-8').split('\n')

    idx = lines.index('"Type","Time(%)","Time","Calls","Avg","Min","Max","Name"')
    pct = float(lines[idx + 2].split(',')[1]) / 100
    ti = float(lines[idx + 2].split(',')[4])

    return ti / pct

def figure6() -> None:
    print('{s:{c}^{n}}'.format(s = 'Experiment Figure6', c = '=', n = '100'))
    model_list = [
        'vgg16', 
        'vgg19', 
        'resnet50', 
        'resnet152', 
        'densenet121', 
        'densenet201', 
        'inceptionv3', 
        'xception', 
        'mobilenet', 
        'efficientnetb0', 
        'MediumCNN 1*1', 
        'MediumCNN 3*3', 
        'MediumCNN 5*5', 
        'MediumCNN 7*7'
    ]

    with open('./results/figure6.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Overhead'])

        for model_name in model_list:
            command_line = 'nvprof --csv python -m src.training_script.overhead_test --model_name {} --batch_size 128 --data_dir {}'.format(model_name, args.imagenet_data_dir)
            p = subprocess.Popen(shlex.split(command_line), stderr=subprocess.PIPE)
            p.wait()
            exec_time = get_execution_time(p.stderr.read())

            command_line_deterministic_tf = 'nvprof --csv python -m src.training_script.overhead_test --model_name {} --batch_size 128 --data_dir {} --deterministic_tf'.format(model_name, args.imagenet_data_dir)
            p = subprocess.Popen(shlex.split(command_line_deterministic_tf), stderr=subprocess.PIPE)
            p.wait()
            exec_time_deterministic = get_execution_time(p.stderr.read())

            writer.writerow([model_name, exec_time_deterministic / exec_time])
    print('{s:{c}^{n}}'.format(s = 'Done', c = '=', n = '100'))



print('List of experiments to perform: ' + ' '.join(args.experiments))

train_models()

if not os.path.exists('./results'):
    os.makedirs('./results')

if 'Figure1' in args.experiments:
    figure1()

if 'Figure2' in args.experiments:
    figure2()

if 'Figure5' in args.experiments:
    figure5()

if 'Figure6' in args.experiments:
    figure6()
