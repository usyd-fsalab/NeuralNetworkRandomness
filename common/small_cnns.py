import tensorflow as tf

def SmallCNN(kernel_size):
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    x = inputs
    channel = 16
    for _ in range(6):
        x = tf.keras.layers.Conv2D(channel, (kernel_size, kernel_size), padding='same')(x)
        channel = channel * 2
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.MaxPooling2D(2, strides=2)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(1000)(x)
    outputs = tf.keras.layers.Activation('softmax', dtype=tf.float32)(x)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)
