import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--ckpt_folder', type=str, required=True)
parser.add_argument('--deterministic_algo', action='store_true')
parser.add_argument('--deterministic_impl', action='store_true')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--tpu', action='store_true')
parser.add_argument('--tpu_zone', type=str, default=None)
parser.add_argument('--tpu_project', type=str, default=None)
parser.add_argument('--tpu_address', type=str, default=None)
parser.add_argument('--test', action='store_true')
parser.add_argument('--data_dir', type=str, default=None, help='')

args = parser.parse_args()

import tensorflow as tf
import tensorflow_datasets as tfds
from ..common.resnet import ResNet18
import os


if not args.tpu:
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

if args.deterministic_impl:
    print('Enabling deterministic tensorflow operations and cuDNN...')
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

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

# if (not args.tpu) and not os.path.exists(args.ckpt_folder):
#     os.makedirs(args.ckpt_folder)
tf.io.gfile.makedirs(args.ckpt_folder)

def celebA_transform(data):
    image = tf.image.resize(data['image'], (128, 128))
    label = []
    for attr in data['attributes']:
        label.append(data['attributes'][attr])
    label = tf.convert_to_tensor(label)
    return image, label

def get_celeba_input(dataset, batch_size, shuffle_seed):
    train_loader = (dataset['train'].map(celebA_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .shuffle(1000, seed=shuffle_seed)
                .batch(batch_size, drop_remainder=args.tpu)
                .prefetch(tf.data.experimental.AUTOTUNE))
    test_loader = (dataset['test'].map(celebA_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE))
    if args.test:
        train_loader = train_loader.take(10)
        test_loader = test_loader.take(10)
    return train_loader, test_loader

def save_model(epoch, logs):
    save_options = None
    if args.tpu:
        save_options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
        tf.keras.models.save_model(model, os.path.join(args.ckpt_folder, f'ckpt{epoch}.h5'), include_optimizer=False, options=save_options)

def lr_scheduler(epoch):
    new_lr = args.lr * (0.1 ** (epoch // 5))
    print('new lr:%.2e' % new_lr)
    return new_lr

if args.deterministic_algo:
    algo_seed = 1
else:
    algo_seed = None

dataset = tfds.load('celeb_a', data_dir=args.data_dir)

train_loader, test_loader = get_celeba_input(dataset, args.batch_size, algo_seed)


model = ResNet18(classes=40, input_shape=(128, 128, 3), seed=algo_seed, weight_decay=0, activation='sigmoid')
optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
model_checkpoint = tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model)
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(args.ckpt_folder, 'log.csv'))
loss = tf.keras.losses.BinaryCrossentropy()
binary_acc = tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=0.5)
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
model.compile(optimizer=optimizer, loss=loss, metrics=[binary_acc])
callbacks = [reduce_lr, model_checkpoint]
if not args.tpu:
    callbacks.append(csv_logger)

model.fit(train_loader, epochs=args.epochs, validation_data=test_loader, callbacks=callbacks)