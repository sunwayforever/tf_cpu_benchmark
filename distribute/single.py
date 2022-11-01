#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2022-11-03 10:02

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(2)

import json
import os
from common import *

train_datasets = get_datasets()
model = get_model()
model.fit(x=train_datasets, epochs=2, steps_per_epoch=5)
