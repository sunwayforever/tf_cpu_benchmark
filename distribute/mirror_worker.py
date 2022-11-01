#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2022-11-03 10:02

import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

import json
import os
import sys
from common import * 

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': int(sys.argv[1])}
})
strategy = tf.distribute.MultiWorkerMirroredStrategy()

train_datasets = get_datasets()
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
train_datasets = train_datasets.with_options(options)

with strategy.scope():
  multi_worker_model = get_model()

multi_worker_model.fit(x=train_datasets, epochs=2, steps_per_epoch=5)
