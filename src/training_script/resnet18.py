from __future__ import print_function

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--deterministic_input', action='store_true')
parser.add_argument('--deterministic_init', action='store_true')
parser.add_argument('--ckpt_folder', type=str, required=True)
parser.add_argument('--tpu', action='store_true')
parser.add_argument('--tpu_zone', type=str, default=None)
parser.add_argument('--tpu_project', type=str, default=None)
parser.add_argument('--tpu_address', type=str, default=None)
parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
parser.add_argument('--deterministic_tf', action='store_true')
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--l2', type=float, default=0.0)
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--fast', action='store_true', help='Fast execution. Used to verify functional instead of correctness')

args = parser.parse_args()

import tensorflow as tf
from tqdm import tqdm
import tensorflow_datasets as tfds
from ..common.resnet import ResNet18
import numpy as np
import time
import argparse
import os
from ..common import input_pipeline

tf.io.gfile.makedirs(args.ckpt_folder)
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

if args.tpu:
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_bfloat16')
    mixed_precision.set_global_policy(policy)
    print('Use Bfloat16')
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + args.tpu_address or os.environ['COLAB_TPU_ADDR'], zone=args.tpu_zone, project=args.tpu_project)
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    strategy = tf.distribute.TPUStrategy(resolver)
    
if args.deterministic_tf:
    print('Enabling deterministic tensorflow operations and cuDNN...')
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

if args.dataset == 'cifar10':
    nb_classes = 10
elif args.dataset == 'cifar100':
    nb_classes = 100

import math
steps_per_epoch = {
    'cifar10' : math.ceil(50000 / args.batch_size),
    'cifar100' : math.ceil(50000 / args.batch_size)
}

nb_epoch = args.epochs
learning_rate = args.lr
# input image dimensions
img_rows, img_cols = 32, 32
if args.dataset == 'imagenet2012':
    img_rows, img_cols = 64, 64
# The CIFAR10 images are RGB.
img_channels = 3

init_seed = None
shuffle_seed = None
np.random.seed(0)
if args.deterministic_init:
    init_seed = 0
if args.deterministic_input:
    shuffle_seed = 0

if args.tpu:
    with strategy.scope():
        model = ResNet18(classes=nb_classes, input_shape=(img_rows, img_cols, img_channels), seed=init_seed, weight_decay=args.l2)
else:
    model = ResNet18(classes=nb_classes, input_shape=(img_rows, img_cols, img_channels), seed=init_seed, weight_decay=args.l2)

train_data, test_data = input_pipeline.get_data(args.dataset, args.batch_size, shuffle_seed=shuffle_seed, tpu=args.tpu, data_dir=args.data_dir)

if args.fast:
    train_data = train_data.take(1)
    test_data = test_data.take(1)
    args.epochs = 1

if args.tpu:
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

def lr_scheduler(epoch):
    new_lr = learning_rate * (0.1 ** (epoch // 50))
    print('new lr:%.2e' % new_lr)
    return new_lr

def save_prediction(epoch, logs):
    if epoch == 9 or (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
        pred_array = np.array([]).reshape(0, nb_classes)
        for x, y in test_data:
            pred = model(x)
            pred_array = np.concatenate((pred_array, pred))

        pred_array = np.argmax(pred_array, axis=1)
        with open(os.path.join(args.ckpt_folder, f'pred{epoch}.txt'), 'w') as f:
            f.write('\n'.join(map(lambda x: str(x), pred_array)))

def save_model(epoch, logs):
    save_options = None
    if args.tpu:
        save_options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    if epoch == 9 or (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
        tf.keras.models.save_model(model, os.path.join(args.ckpt_folder, f'ckpt{epoch}.h5'), include_optimizer=False, options=save_options)


save_pred_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=save_prediction, verbose=True)
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(args.ckpt_folder, 'log.csv'))
model_checkpoint = tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model)

####################################### TODO no lr decay here
callbacks = [reduce_lr, model_checkpoint, save_pred_callback]
if not args.tpu:
    callbacks.append(csv_logger)

model.fit(train_data, epochs=args.epochs, callbacks=callbacks, validation_data=test_data, shuffle=False, steps_per_epoch=steps_per_epoch[args.dataset])

###########################################