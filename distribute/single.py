#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2022-11-03 10:02

import tensorflow as tf
from common import *

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(get_num_workers())

train_datasets = get_datasets()
model = get_model()
model.fit(x=train_datasets, epochs=1, steps_per_epoch=5)
