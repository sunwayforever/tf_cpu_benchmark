#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2022-11-03 11:17
import tensorflow as tf
def get_model():
    model = tf.keras.models.Sequential(
        [
            # tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Reshape((28, 28, 1)),
            tf.keras.layers.Conv2D(128, 5, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(128, 5, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(128, 5, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(128, 5, padding="same", activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
        ]
    )

    model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
    return model

import tensorflow_datasets as tfds
tfds.disable_progress_bar()
BUFFER_SIZE = 10000
NUM_WORKERS = 2
GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS

def get_datasets():
  # 将 MNIST 数据从 (0, 255] 缩放到 (0., 1.]
  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

  datasets, info = tfds.load(name='mnist',
                            with_info=True,
                            as_supervised=True)

  return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
