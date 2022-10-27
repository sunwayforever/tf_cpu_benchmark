#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2022-10-13 22:13
import tensorflow as tf
import sys
import time

# workaround: disable grappler
opts = tf.config.optimizer.get_experimental_options()
opts["disable_meta_optimizer"] = True
tf.config.optimizer.set_experimental_options(opts)
inter = int(sys.argv[1])
intra = int(sys.argv[2])
model_name = sys.argv[3]
model_p1 = int(sys.argv[4])
model_p2 = int(sys.argv[5])

tf.config.threading.set_inter_op_parallelism_threads(inter)
tf.config.threading.set_intra_op_parallelism_threads(intra)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[:100]
y_train = y_train[:100]
x_test = x_test[:100]
y_test = y_test[:100]


def get_dscnn(a, b):
    model = tf.keras.models.Sequential(
        [
            # tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Reshape((28, 28, 1)),
            tf.keras.layers.SeparableConv2D(a, b, padding="same", activation="relu"),
            tf.keras.layers.SeparableConv2D(a, b, padding="same", activation="relu"),
            tf.keras.layers.SeparableConv2D(a, b, padding="same", activation="relu"),
            tf.keras.layers.SeparableConv2D(a, b, padding="same", activation="relu"),
            tf.keras.layers.SeparableConv2D(a, b, padding="same", activation="relu"),
            tf.keras.layers.SeparableConv2D(a, b, padding="same", activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
        ]
    )
    return model


def get_cnn(a, b):
    model = tf.keras.models.Sequential(
        [
            # tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Reshape((28, 28, 1)),
            tf.keras.layers.Conv2D(a, b, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(a, b, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(a, b, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(a, b, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(a, b, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(a, b, padding="same", activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
        ]
    )
    return model


def get_inception(a, b):
    input = tf.keras.Input(shape=(28, 28))
    output = tf.keras.layers.Reshape((28, 28, 1))(input)
    p1 = [
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
    ]
    output = tf.keras.layers.Concatenate()(p1)
    p2 = [
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
        tf.keras.layers.Conv2D(a, b, padding="same", activation="relu")(output),
    ]
    output = tf.keras.layers.Concatenate()(p2)
    output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Flatten()(output)
    output = tf.keras.layers.Dense(10)(output)
    return tf.keras.Model(inputs=input, outputs=output)


def get_dnn(a):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(a, activation="relu"),
            tf.keras.layers.Dense(a, activation="relu"),
            tf.keras.layers.Dense(a, activation="relu"),
            tf.keras.layers.Dense(a, activation="relu"),
            tf.keras.layers.Dense(a, activation="relu"),
            tf.keras.layers.Dense(a, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )
    return model


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model = None
if model_name == "dnn":
    model = get_dnn(model_p1)

if model_name == "cnn":
    model = get_cnn(model_p1, model_p2)

if model_name == "dscnn":
    model = get_dscnn(model_p1, model_p2)

if model_name == "inception":
    model = get_inception(model_p1, model_p2)

model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

a = time.time()
b = time.process_time()
model.fit(x_train, y_train, epochs=1, verbose=0)  # , callbacks = [tb]
print(
    f"{model_name}:{model_p1}:{model_p2}:{inter}:{intra}:{int((time.time() - a) * 1000)}:{int((time.process_time() - b) * 1000)}"
)
