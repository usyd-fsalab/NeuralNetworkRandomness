import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='')
parser.add_argument('--deterministic_init', action='store_true', help='')
parser.add_argument('--deterministic_input', action='store_true', help='')
parser.add_argument('--deterministic_tf', action='store_true', help='')
parser.add_argument('--ckpt_folder', type=str, required=True, help='')
parser.add_argument('--lr', type=float, required=True, help='')
parser.add_argument('--epochs', type=int, default=200, help='')
parser.add_argument('--tpu', action='store_true', help='')
parser.add_argument('--tpu_zone', type=str, default=None, help='')
parser.add_argument('--tpu_project', type=str, default=None, help='')
parser.add_argument('--tpu_address', type=str, default=None, help='')
parser.add_argument('--fp16', action='store_true', help='')
parser.add_argument('--bn', action='store_true', help='')
parser.add_argument('--data_dir', type=str, default=None, help='')

args = parser.parse_args()

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras.initializers import GlorotUniform
import os

import time
import numpy as np

# if (not args.tpu) and (not os.path.exists(args.ckpt_folder)):
#     os.makedirs(args.ckpt_folder)
tf.io.gfile.makedirs(args.ckpt_folder)

physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

if args.deterministic_tf:
  print('Enabling deterministic tensorflow operations and cuDNN...')
  os.environ["TF_DETERMINISTIC_OPS"] = "1"
  os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

if args.fp16:
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.experimental.set_policy(policy)
    print('Use Tensor Cores.')
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)

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

dataset = tfds.load('cifar10', data_dir=args.data_dir)


trainloader, testloader = dataset['train'].cache(), dataset['test'].cache()

AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = args.batch_size
IMG_SHAPE = 32



def preprocess_image(data):
  img = tf.cast(data['image'], tf.float32)
  img = img/255.

  return img, tf.one_hot(data['label'], 10)

if args.deterministic_input:
    trainloader = (
        trainloader
        .shuffle(1024, seed=0)
        .map(preprocess_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE, drop_remainder=args.tpu)
        .prefetch(AUTO)
    )
else:
    trainloader = (
        trainloader
        .shuffle(1024)
        .map(preprocess_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE, drop_remainder=args.tpu)
        .prefetch(AUTO)
    )

testloader = (
    testloader
    .map(preprocess_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

def Model():
  inputs = keras.layers.Input(shape=(IMG_SHAPE, IMG_SHAPE, 3))
  if args.deterministic_init:
    initializer = GlorotUniform(seed=0)
  else:
    initializer = GlorotUniform()

  x = keras.layers.Conv2D(16, (3,3), padding='same', kernel_initializer=initializer)(inputs)
  if args.bn:
    x = keras.layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = keras.layers.MaxPooling2D(2, strides=2)(x)

  x = keras.layers.Conv2D(32,(3,3), padding='same', kernel_initializer=initializer)(x)
  if args.bn:
    x = keras.layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = keras.layers.MaxPooling2D(2, strides=2)(x)

  x = keras.layers.Conv2D(32,(3,3), padding='same', kernel_initializer=initializer)(x)
  if args.bn:
    x = keras.layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = keras.layers.MaxPooling2D(2, strides=2)(x)

  x = keras.layers.GlobalAveragePooling2D()(x)
  x = keras.layers.Dense(32, activation='relu', kernel_initializer=initializer)(x)
  # x = keras.layers.Dropout(0.1)(x)
  
  x = keras.layers.Dense(10, kernel_initializer=initializer)(x)
  outputs = tf.keras.layers.Activation('softmax', dtype=tf.float32)(x)

  return keras.models.Model(inputs=inputs, outputs=outputs)


tf.keras.backend.clear_session()
model = Model()


def lr_schedule(epoch):
  new_lr = 0
  if epoch < 50:
    new_lr = args.lr
  elif (epoch >=50) & (epoch < 100):
    new_lr = args.lr / 10
  elif epoch >= 100:
    new_lr = args.lr / 100

  return new_lr

def save_prediction(epoch, logs):
  if epoch == 9 or (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
    pred_array = np.array([]).reshape(0, 10)
    for x, y in testloader:
      pred = model(x)
      pred_array = np.concatenate((pred_array, pred))

    pred_array = np.argmax(pred_array, axis=1)
    with open(os.path.join(args.ckpt_folder, f'pred{epoch}.txt'), 'w') as f:
      f.write('\n'.join(map(lambda x: str(x), pred_array)))

    # tf.io.write_file(, )

def save_model(epoch, logs):
  save_options = None
  if args.tpu:
      save_options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
  if epoch == 9 or (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
    tf.keras.models.save_model(model, os.path.join(args.ckpt_folder, f'ckpt{epoch}.h5'), options=save_options)

save_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=save_prediction, verbose=True)
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(args.ckpt_folder, 'log.csv'))
ckpt_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model)
lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr_schedule(epoch), verbose=True)

keras.backend.clear_session()
if args.tpu:
  with strategy.scope():
    model = Model()
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    # optimizer = keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
else:
  model = Model()
  optimizer = keras.optimizers.Adam(learning_rate=args.lr)
  # optimizer = keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [save_callback, lr_callback, ckpt_callback]

if not args.tpu:
  callbacks.append(csv_logger)

_ = model.fit(trainloader,
          epochs=args.epochs,
          callbacks=callbacks,
          validation_data=testloader)

