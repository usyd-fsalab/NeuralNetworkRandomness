from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import GlorotUniform
# from nvtx.plugins.tf.keras.layers import NVTXStart
# from nvtx.plugins.tf.keras.layers import NVTXEnd
# from .arguments import args

import tensorflow as tf
def conv2d_bn(x, filters, kernel_size, seed, weight_decay=.0, strides=(1, 1)):
    layer = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(weight_decay),
                   kernel_initializer=GlorotUniform(seed=seed)
                   )(x)
    layer = BatchNormalization()(layer)
    return layer


def conv2d_bn_relu(x, filters, kernel_size, seed, weight_decay=.0, strides=(1, 1)):
    layer = conv2d_bn(x, filters, kernel_size=kernel_size, seed=seed, weight_decay=weight_decay, strides=strides)
    layer = Activation('relu')(layer)
    return layer


def ResidualBlock(x, filters, kernel_size, weight_decay, seed, downsample=True):
    if downsample:
        # residual_x = conv2d_bn_relu(x, filters, kernel_size=1, strides=2)
        residual_x = conv2d_bn(x, filters, kernel_size=1, seed=seed, weight_decay=weight_decay, strides=2)
        # input_channel = x.shape[-1]
        # residual_x = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='VALID')
        # residual_x = tf.pad(residual_x, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
        #                                                              input_channel // 2]])
        stride = 2
    else:
        residual_x = x
        stride = 1
    residual = conv2d_bn_relu(x,
                              filters=filters,
                              kernel_size=kernel_size,
                              seed=seed,
                              weight_decay=weight_decay,
                              strides=stride,
                              )
    residual = conv2d_bn(residual,
                         filters=filters,
                         kernel_size=kernel_size,
                         seed=seed,
                         weight_decay=weight_decay,
                         strides=1,
                         )
    out = layers.add([residual_x, residual])
    out = Activation('relu')(out)
    # if args.dropout:
    #     print('Use mc-dropout')
    #     out = tf.keras.layers.Dropout(rate=0.1)(out, training=True)
    return out

class CustomTrainStepModel(Model):
    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            pred = self(images)
            loss = self.compiled_loss(labels, pred, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(labels, pred)
        return {m.name: m.result() for m in self.metrics}

def ResNet18(classes, input_shape, seed, weight_decay, activation='softmax'):
    input = Input(shape=input_shape)
    x = input
    x = conv2d_bn_relu(x, filters=64, kernel_size=(3, 3), seed=seed, weight_decay=weight_decay, strides=(1, 1))
    # x = MaxPool2D(pool_size=(3, 3), strides=(2, 2),  padding='same')(x)
    # x, marker_id, domain_id = NVTXStart(message='Resnet', trainable=True)(x)
    # x = conv2d_bn_relu(x, filters=64, kernel_size=(3, 3), seed=seed, weight_decay=weight_decay, strides=(1, 1))

    # # conv 2
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), seed=seed, weight_decay=weight_decay, downsample=False)
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), seed=seed, weight_decay=weight_decay, downsample=False)
    # # conv 3
    x = ResidualBlock(x, filters=128, kernel_size=(3, 3), seed=seed, weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=128, kernel_size=(3, 3), seed=seed, weight_decay=weight_decay, downsample=False)
    # # conv 4
    x = ResidualBlock(x, filters=256, kernel_size=(3, 3), seed=seed, weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=256, kernel_size=(3, 3), seed=seed, weight_decay=weight_decay, downsample=False)
    # # conv 5
    x = ResidualBlock(x, filters=512, kernel_size=(3, 3), seed=seed, weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=512, kernel_size=(3, 3), seed=seed, weight_decay=weight_decay, downsample=False)
    x = AveragePooling2D(pool_size=(4, 4), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(classes, kernel_initializer=GlorotUniform(seed=seed))(x)

    x = tf.keras.layers.Activation(activation, dtype='float32', name='predictions')(x)

    # x = NVTXEnd(grad_message='Resnet Grad')([x, marker_id, domain_id])
    model = Model(input, x, name='ResNet18')

    return model