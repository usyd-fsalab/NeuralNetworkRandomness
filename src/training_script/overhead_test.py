import tensorflow as tf
import tensorflow_datasets as tfds
from ..common.cnns import MediumCNN
import argparse
import os 
parser = argparse.ArgumentParser()
parser.add_argument('--deterministic_tf', action='store_true')
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--data_dir', type=str, required=True)
args = parser.parse_args()

physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

if args.deterministic_tf:
    print('Enabling deterministic tensorflow operations and cuDNN...')
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

def get_model(model_name, input_shape=(224, 224, 3), num_classes=1000):
    if model_name == 'resnet50':
        return tf.keras.applications.ResNet50(weights=None, input_shape=input_shape, classes=num_classes)
    if model_name == 'resnet101':
        return tf.keras.applications.ResNet101(weights=None, input_shape=input_shape, classes=num_classes)
    if model_name == 'resnet152':
        return tf.keras.applications.ResNet152(weights=None, input_shape=input_shape, classes=num_classes)
    if model_name == 'inceptionv3':
        return tf.keras.applications.InceptionV3(weights=None, input_shape=input_shape, classes=num_classes)
    if model_name == 'xception':
        return tf.keras.applications.Xception(weights=None, input_shape=input_shape, classes=num_classes)
    if model_name == 'densenet121':
        return tf.keras.applications.DenseNet121(weights=None, input_shape=input_shape, classes=num_classes)
    if model_name == 'densenet169':
        return tf.keras.applications.DenseNet169(weights=None, input_shape=input_shape, classes=num_classes)
    if model_name == 'densenet201':
        return tf.keras.applications.DenseNet201(weights=None, input_shape=input_shape, classes=num_classes)
    if model_name == 'mobilenet':
        return tf.keras.applications.MobileNet(weights=None, input_shape=input_shape, classes=num_classes)
    if model_name == 'vgg16':
        return tf.keras.applications.VGG16(weights=None, input_shape=input_shape, classes=num_classes)
    if model_name == 'vgg19':
        return tf.keras.applications.VGG19(weights=None, input_shape=input_shape, classes=num_classes)
    if model_name == 'efficientnetb0':
        return tf.keras.applications.EfficientNetB0(weights=None, input_shape=input_shape, classes=num_classes)
    if model_name == 'efficientnetb1':
        return tf.keras.applications.EfficientNetB1(weights=None, input_shape=input_shape, classes=num_classes)
    if model_name == 'efficientnetb2':
        return tf.keras.applications.EfficientNetB2(weights=None, input_shape=input_shape, classes=num_classes)
    if model_name == 'efficientnetb4':
        return tf.keras.applications.EfficientNetB4(weights=None, input_shape=input_shape, classes=num_classes)
    if model_name == 'efficientnetb7':
        return tf.keras.applications.EfficientNetB7(weights=None, input_shape=input_shape, classes=num_classes)
    if model_name == 'MediumCNN 1*1':
        return MediumCNN(1)
    if model_name == 'MediumCNN 3*3':
        return MediumCNN(3)
    if model_name == 'MediumCNN 5*5':
        return MediumCNN(5)
    if model_name == 'MediumCNN 7*7':
        return MediumCNN(7)
    raise Exception('Model Not Implemented')

def pre_process(data):
    image = tf.image.resize(data['image'], [224, 224])
    label = tf.one_hot(data['label'], depth=1000)
    return image, label

model = get_model(args.model_name)
model.compile(optimizer='adam', loss='categorical_crossentropy')

train_loader = tfds.load('imagenet2012', data_dir=args.data_dir)['train']


train_loader = train_loader.map(pre_process, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE).take(100)

model.fit(train_loader, epochs=1, verbose=0)


