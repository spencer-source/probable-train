
import math
import tensorflow as tf
import tensorflow_addons as tfa

from collections.abc import Iterable


def make_model(input_shape):
    ip = tf.keras.Input([128, 1])
    x = WaveletBlock(4)(ip)
    x = tf.keras.layers.Conv1D(256, 1, activation="gelu")(x)
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    y = tf.keras.layers.Dense(9, activation="softmax")(x)
    model = tf.keras.Model(ip, y)
    return model

def get_compiled_model(input_shape):
    model = make_model(input_shape)
    
    model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    return model

model = get_compiled_model([128, 1])

training = group_padded(train_ds, 64)
validation = group_padded(val_ds, 64)
testing = group_padded(test_ds, 64)

model.fit(training, validation_data=validation, epochs=5)
model.evaluate(testing)
