import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

num_classes = None


def horizontal_flip(image, axis):
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        assert axis == 0 or axis == 1
        # image = cv2.flip(image, axis)
        image = np.flip(image, axis)
    
    return image

def cifar_random_crop_and_flip(image):

    image = np.pad(image.numpy(), ((4, 4), (4, 4), (0, 0)), mode='constant', constant_values=0)
    x_offset = np.random.randint(low=0, high=8, size=1)[0]
    y_offset = np.random.randint(low=0, high=8, size=1)[0]
    
    image = image[x_offset:x_offset+32,y_offset:y_offset+32,:]
    image = horizontal_flip(image=image, axis=1)
    return image


def cifar_data_preprocessing(data):
    image = tf.cast(data['image'], tf.float32)
    image = tf.image.per_image_standardization(image)
    return image, tf.one_hot(data['label'], depth=num_classes)

def cifar_data_aug(image, label):
    # image = tf.py_function(cifar_random_crop_and_flip, [image], tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, [32, 32, 3], seed=1)
    image = tf.image.random_flip_left_right(image, seed=2)
    image = tf.convert_to_tensor(image)
    image.set_shape([32, 32, 3])
    return image, label

def get_cifar_input(dataset, batch_size, shuffle_seed, tpu):
    train_loader = dataset['train'].map(cifar_data_preprocessing).cache().repeat().map(cifar_data_aug).shuffle(10000, seed=shuffle_seed).batch(batch_size, drop_remainder=tpu).prefetch(tf.data.experimental.AUTOTUNE)
    test_loader = dataset['test'].map(cifar_data_preprocessing).cache().batch(200).prefetch(tf.data.experimental.AUTOTUNE)
    return train_loader, test_loader


def get_data(dataset_name, batch_size, shuffle_seed, tpu=False, data_dir=None):
    global num_classes

    dataset = tfds.load(dataset_name, data_dir=data_dir)

    if dataset_name == 'cifar10':
        num_classes = 10
        return get_cifar_input(dataset, batch_size, shuffle_seed, tpu)
    elif dataset_name == 'cifar100':
        num_classes = 100
        return get_cifar_input(dataset, batch_size, shuffle_seed, tpu)
    else:
        raise Exception(f'Unexpedted dataset {dataset_name}')