# Copyright 2018 Lukas Jendele and Ondrej Skopek. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from datetime import datetime
import os

import keras
from keras import backend as K

import numpy as np

import tensorflow as tf


def set_seeds(seed=42, graph=None):
    import os
    np.random.seed(seed=seed)
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if graph is not None:
        graph.seed = seed


# Make sure to implement the run() function in the model file!
class BaseModel:

    def __init__(self, logdir_name="logs", checkpoint_dirname="checkpoints", expname="exp", threads=1, seed=42):
        self.expname = expname

        # Create an empty graph and a session
        graph = tf.Graph()
        set_seeds(seed=seed, graph=graph)
        self.session = tf.Session(
            graph=graph,
            config=tf.ConfigProto(inter_op_parallelism_threads=threads, intra_op_parallelism_threads=threads))

        # Construct the graph
        with self.session.graph.as_default():
            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.is_training = tf.placeholder_with_default(False, [])

            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.logdir = "{}/{}-{}".format(logdir_name, expname, timestamp)
            self.saver = tf.train.Saver(max_to_keep=10)
            self.summary_writer = tf.summary.FileWriter(self.logdir, flush_secs=5 * 1000)
            self.save_path = os.path.join(self.logdir, checkpoint_dirname)
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

    def _init_variables(self):
        # Initialize variables
        self.session.run(tf.global_variables_initializer())

    @property
    def training_step(self):
        return self.session.run(self.global_step)


class KerasModel(object):

    callbacks = []

    def __init__(self, logdir_name="logs", checkpoint_dirname="checkpoints", expname="exp", threads=1, seed=42):
        set_seeds(seed=seed)
        session = tf.Session(
            config=tf.ConfigProto(inter_op_parallelism_threads=threads, intra_op_parallelism_threads=threads))
        K.set_session(session)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.logdir = "{}/{}-{}".format(logdir_name, expname, timestamp)
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=self.logdir, histogram_freq=1, write_graph=True, write_images=True)
        self.callbacks.append(tensorboard)

        self.save_path = os.path.join(self.logdir, checkpoint_dirname)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        checkpoints = keras.callbacks.ModelCheckpoint(
            os.path.join(self.save_path, "model.hdf5"),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            period=1)
        self.callbacks.append(checkpoints)
