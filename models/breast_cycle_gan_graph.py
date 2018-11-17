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

import os

import numpy as np
import tensorflow as tf
import random

from models.base import BaseModel
from resources.data.utils import next_batch, shuffle
from resources.model_utils import tile_images

# Flags
from flags import flags_parser
FLAGS = flags_parser.FLAGS
assert FLAGS is not None

BATCH_SIZE = 1
ngf = 32
ndf = 64


# Copied from Cycle-GAN implmentation
def instance_norm(x):
    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable(
            'scale', [x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [x.get_shape()[-1]], initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

        return out


def conv2d(x, filters=3, kernel=3, strides=1, padding='VALID', relu=0.2, norm=True, name='conv'):
    with tf.variable_scope(name):
        out_res = tf.layers.conv2d(
            x,
            filters=filters,
            kernel_size=kernel,
            strides=strides,
            padding=padding,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        if norm:
            out_res = instance_norm(out_res)
        if relu > 0:
            out_res = tf.nn.leaky_relu(out_res, alpha=relu)
        elif relu == 0:
            out_res = tf.nn.relu(out_res)
        return out_res


def deconv2d(x, outshape, filters=64, kernel=7, strides=1, padding="VALID", name="deconv2d", norm=True, relu=0.2):
    with tf.variable_scope(name):

        conv = tf.layers.conv2d_transpose(
            x,
            filters=filters,
            kernel_size=kernel,
            strides=strides,
            padding=padding,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

        if norm:
            conv = instance_norm(conv)

        if relu == 0:
            conv = tf.nn.relu(conv, "relu")
        elif relu > 0:
            conv = tf.nn.leaky_relu(conv, alpha=relu)

        return conv


def build_resnet_block(x, dim, name="resnet", padding='REFLECT'):
    with tf.variable_scope(name):
        with tf.variable_scope("Conv1"):
            out_res = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
            out_res = conv2d(out_res, filters=dim, name="conv1")
        with tf.variable_scope("Conv2"):
            out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
            out_res = conv2d(out_res, filters=dim, name="conv2", relu=-1)

        return tf.nn.relu(out_res + x)


def build_generator_resnet_9blocks(x, name="generator", skip=True):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "CONSTANT"

        pad_input = tf.pad(x, [[0, 0], [ks, ks], [ks, ks], [0, 0]], padding)
        o_c1 = conv2d(pad_input, filters=ngf, kernel=f, name="conv1")
        o_c2 = conv2d(o_c1, filters=ngf * 2, kernel=ks, strides=2, padding="SAME", name="conv2")
        o_c3 = conv2d(o_c2, filters=ngf * 4, kernel=ks, strides=2, padding="SAME", name="conv3")

        o_r = build_resnet_block(o_c3, ngf * 4, "r1", padding)
        for i in range(2, 10):
            o_r = build_resnet_block(o_r, ngf * 4, name="r" + str(i), padding=padding)

        o_c4 = deconv2d(
            o_r, [BATCH_SIZE, 128, 128, ngf * 2], filters=ngf * 2, kernel=ks, strides=2, padding="SAME", name="dc4")
        o_c5 = deconv2d(
            o_c4, [BATCH_SIZE, 256, 256, ngf], filters=ngf, kernel=ks, strides=2, padding="SAME", name="dc5")
        o_c6 = conv2d(o_c5, filters=1, kernel=f, padding="SAME", name="c6", norm=False, relu=-1)

        if skip is True:
            out_gen = tf.nn.tanh(x + o_c6, "t1")
        else:
            out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen


def discriminator(x, name="discriminator"):
    with tf.variable_scope(name):
        f = 4

        padw = 2

        with tf.variable_scope('Conv1'):
            pad_input = tf.pad(x, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
            o_c1 = conv2d(pad_input, filters=ndf, kernel=f, strides=2, name="c1", norm=False)

        with tf.variable_scope('Conv2'):
            pad_o_c1 = tf.pad(o_c1, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
            o_c2 = conv2d(pad_o_c1, filters=ndf * 2, kernel=f, strides=2, name="c2")

        with tf.variable_scope('Conv3'):
            pad_o_c2 = tf.pad(o_c2, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
            o_c3 = conv2d(pad_o_c2, filters=ndf * 4, kernel=f, strides=2, name="c3")

        with tf.variable_scope('Conv4'):
            pad_o_c3 = tf.pad(o_c3, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
            o_c4 = conv2d(pad_o_c3, filters=ndf * 8, kernel=f, strides=1, name="c4")

        with tf.variable_scope('Conv5'):
            pad_o_c4 = tf.pad(o_c4, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
            o_c5 = conv2d(pad_o_c4, filters=1, kernel=f, name="c5", norm=False, relu=-1)

        return o_c5


def patch_discriminator(x, name="discriminator"):
    with tf.variable_scope(name):
        f = 4

        patch_input = tf.random_crop(x, [1, 70, 70, 3])
        o_c1 = conv2d(patch_input, filters=ndf, kernel=f, strides=2, padding="SAME", name="c1", norm=False)
        o_c2 = conv2d(o_c1, filters=ndf * 2, kernel=f, strides=2, padding="SAME", name="c2")
        o_c3 = conv2d(o_c2, filters=ndf * 4, kernel=f, strides=2, padding="SAME", name="c3")
        o_c4 = conv2d(o_c3, filters=ndf * 8, kernel=f, strides=2, padding="SAME", name="c4")
        o_c5 = conv2d(o_c4, filters=1, kernel=f, padding="SAME", name="c5", norm=False, relu=-1)

        return o_c5


def build_model(inputs, skip=True):
    images_a = inputs['images_a']
    images_b = inputs['images_b']

    fake_pool_a = inputs['fake_pool_a']
    fake_pool_b = inputs['fake_pool_b']

    with tf.variable_scope("Model") as scope:
        prob_real_a_is_real = discriminator(images_a, "d_A")
        prob_real_b_is_real = discriminator(images_b, "d_B")
        generator = build_generator_resnet_9blocks

        fake_images_b = generator(images_a, name="g_A", skip=skip)
        fake_images_a = generator(images_b, name="g_B", skip=skip)

        scope.reuse_variables()

        prob_fake_a_is_real = discriminator(fake_images_a, "d_A")
        prob_fake_b_is_real = discriminator(fake_images_b, "d_B")

        cycle_images_a = generator(fake_images_b, "g_B", skip=skip)
        cycle_images_b = generator(fake_images_a, "g_A", skip=skip)

        scope.reuse_variables()

        prob_fake_pool_a_is_real = discriminator(fake_pool_a, "d_A")
        prob_fake_pool_b_is_real = discriminator(fake_pool_b, "d_B")

    return {
        'prob_real_a_is_real': prob_real_a_is_real,
        'prob_real_b_is_real': prob_real_b_is_real,
        'prob_fake_a_is_real': prob_fake_a_is_real,
        'prob_fake_b_is_real': prob_fake_b_is_real,
        'prob_fake_pool_a_is_real': prob_fake_pool_a_is_real,
        'prob_fake_pool_b_is_real': prob_fake_pool_b_is_real,
        'cycle_images_a': cycle_images_a,
        'cycle_images_b': cycle_images_b,
        'fake_images_a': fake_images_a,
        'fake_images_b': fake_images_b,
    }


# Shortcut for loss calculation.
def cycle_consistency_loss(real_images, generated_images):
    """L1-norm difference between the real and generated images."""
    return tf.reduce_mean(tf.abs(real_images - generated_images))


def lsgan_loss_generator(prob_fake_is_real):
    """Least-squares generator loss"""
    return tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 1))


def lsgan_loss_discriminator(prob_real_is_real, prob_fake_is_real):
    """Least-squares discriminator losses."""
    return (tf.reduce_mean(tf.squared_difference(prob_real_is_real, 1)) + tf.reduce_mean(
        tf.squared_difference(prob_fake_is_real, 0))) * 0.5


def cross_entropy_loss(logits=None, labels=None):
    # return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    return tf.reduce_mean(logits)


def read_small(filename):
    images = np.load(filename)
    healthy = images["healthy"]
    cancer = images["cancer"]
    healthy = np.reshape(healthy, (-1, 256, 256))
    cancer = np.reshape(cancer, (-1, 256, 256))
    healthy = np.expand_dims(healthy, -1)
    cancer = np.expand_dims(cancer, -1)
    return healthy, cancer


# Model
class CycleGan(BaseModel):
    # Setup constants
    IMAGE_SIZE = 28
    NEW_IMAGE_SIZE = 256
    IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
    NOISE_SIZE = 100

    def __init__(self):
        super(CycleGan, self).__init__(
            logdir_name=FLAGS.data.out_dir,
            checkpoint_dirname=FLAGS.training.checkpoint_dir,
            expname="Cycle-GAN",
            threads=FLAGS.training.threads,
            seed=FLAGS.training.seed)
        with self.session.graph.as_default():
            self._build()
            self._init_variables()

            self.fake_images_A = np.zeros((FLAGS.model.optimization.pool_size, 1, self.NEW_IMAGE_SIZE,
                                           self.NEW_IMAGE_SIZE, 1))
            self.fake_images_B = np.zeros((FLAGS.model.optimization.pool_size, 1, self.NEW_IMAGE_SIZE,
                                           self.NEW_IMAGE_SIZE, 1))
            self.summary_writer = tf.summary.FileWriter(self.logdir, flush_secs=5 * 1000)

    # Construct the graph
    def _build(self):
        self.d_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="d_step")
        self.g_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="g_step")

        self.input_a = tf.placeholder(tf.float32, shape=(None, self.NEW_IMAGE_SIZE, self.NEW_IMAGE_SIZE, 1))
        print(self.input_a.get_shape())
        self.images_input_a = self.input_a  # tf.reshape(self.input_a, (-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1))
        # print(self.images_input_a.get_shape())
        # self.images_input_a = tf.image.resize_images(self.images_input_a, [self.NEW_IMAGE_SIZE, self.NEW_IMAGE_SIZE])
        # print(self.images_input_a.get_shape())
        # self.images_input_a = (self.images_input_a - 0.5) * 2.0
        print(self.images_input_a.get_shape())

        self.input_b = tf.placeholder(tf.float32, shape=(None, self.NEW_IMAGE_SIZE, self.NEW_IMAGE_SIZE, 1))
        print(self.input_b.get_shape())
        self.images_input_b = self.input_b  # tf.reshape(self.input_b, (-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1))
        # print(self.images_input_b.get_shape())
        # self.images_input_b = tf.image.resize_images(self.images_input_b, [self.NEW_IMAGE_SIZE, self.NEW_IMAGE_SIZE])
        # print(self.images_input_b.get_shape())
        # self.images_input_b = (self.images_input_b - 0.5) * 2.0
        print(self.images_input_b.get_shape())

        self.fake_pool_A = tf.placeholder(
            tf.float32, [None, self.NEW_IMAGE_SIZE, self.NEW_IMAGE_SIZE, 1], name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(
            tf.float32, [None, self.NEW_IMAGE_SIZE, self.NEW_IMAGE_SIZE, 1], name="fake_pool_B")

        inputs = {
            'images_a': self.images_input_a,
            'images_b': self.images_input_b,
            'fake_pool_a': self.fake_pool_A,
            'fake_pool_b': self.fake_pool_B
        }

        self.training = tf.placeholder_with_default(False, shape=())
        self.noise_input_interpolated = tf.placeholder(tf.float32, shape=(None, self.NOISE_SIZE))
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")
        self.num_fake_inputs = 0

        self.outputs = outputs = build_model(inputs, skip=True)

        self.prob_real_a_is_real = outputs['prob_real_a_is_real']
        self.prob_real_b_is_real = outputs['prob_real_b_is_real']
        self.fake_images_a = outputs['fake_images_a']
        self.fake_images_b = outputs['fake_images_b']
        self.prob_fake_a_is_real = outputs['prob_fake_a_is_real']
        self.prob_fake_b_is_real = outputs['prob_fake_b_is_real']

        self.cycle_images_a = outputs['cycle_images_a']
        self.cycle_images_b = outputs['cycle_images_b']

        self.prob_fake_pool_a_is_real = outputs['prob_fake_pool_a_is_real']
        self.prob_fake_pool_b_is_real = outputs['prob_fake_pool_b_is_real']

        # Losses
        print(self.images_input_a.get_shape())
        print(self.cycle_images_a.get_shape())
        print(FLAGS.model.optimization.lambda_a)
        cycle_consistency_loss_a = FLAGS.model.optimization.lambda_a * cycle_consistency_loss(
            real_images=self.images_input_a, generated_images=self.cycle_images_a)
        cycle_consistency_loss_b = FLAGS.model.optimization.lambda_b * cycle_consistency_loss(
            real_images=self.images_input_b, generated_images=self.cycle_images_b)

        lsgan_loss_a = lsgan_loss_generator(self.prob_fake_a_is_real)
        lsgan_loss_b = lsgan_loss_generator(self.prob_fake_b_is_real)

        g_loss_A = cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_b
        g_loss_B = cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_a

        d_loss_A = lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_a_is_real, prob_fake_is_real=self.prob_fake_pool_a_is_real)
        d_loss_B = lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_b_is_real, prob_fake_is_real=self.prob_fake_pool_b_is_real)

        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]

        self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)

        # Summary variables for tensorboard
        self.g_A_loss_summ = tf.summary.scalar("loss/generator_A", g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("loss/generator_B", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("loss/discriminator_A", d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("loss/discriminator_B", d_loss_B)

        # Test summaries
        results = tf.stack([
            self.images_input_a, self.fake_images_b, self.cycle_images_a, self.images_input_b, self.fake_images_a,
            self.cycle_images_b
        ],
                           axis=0)
        tiled_image_random = tile_images(results, 3, 2, self.NEW_IMAGE_SIZE, self.NEW_IMAGE_SIZE)
        image_summary_op = tf.summary.image('generated_images', tiled_image_random, max_outputs=1)

        results_diff = tf.stack([
            self.fake_images_b - self.images_input_a, self.cycle_images_a - self.fake_images_b,
            self.fake_images_a - self.images_input_b, self.cycle_images_b - self.fake_images_a
        ],
                                axis=0)
        tiled_image_random_diff = tile_images(results_diff, 2, 2, self.NEW_IMAGE_SIZE, self.NEW_IMAGE_SIZE)
        image_summary_op_diff = tf.summary.image('generated_images_diff', tiled_image_random_diff, max_outputs=1)
        self.gen_image_summary_op = tf.summary.merge([image_summary_op, image_summary_op_diff])

    def fake_image_pool(self, num_fakes, fake, fake_pool):
        """
        This function saves the generated image to corresponding
        pool of images.
        It keeps on feeling the pool till it is full and then randomly
        selects an already stored image and replace it with new one.
        """
        if num_fakes < FLAGS.model.optimization.pool_size:
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, FLAGS.model.optimization.pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake

    def train_batch(self, batch, step, curr_lr):
        # Optimizing the G_A network
        _, fake_B_temp, summary_str = self.session.run([self.g_A_trainer, self.fake_images_b, self.g_A_loss_summ],
                                                       feed_dict={
                                                           self.input_a: batch['images_a'],
                                                           self.input_b: batch['images_b'],
                                                           self.learning_rate: curr_lr
                                                       })
        self.summary_writer.add_summary(summary_str, step)

        fake_B_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_B_temp, self.fake_images_B)

        # Optimizing the D_B network
        _, summary_str = self.session.run(
            [self.d_B_trainer, self.d_B_loss_summ],
            feed_dict={
                self.input_a: batch['images_a'],
                self.input_b: batch['images_b'],
                self.learning_rate: curr_lr,
                self.fake_pool_B: fake_B_temp1
            })
        self.summary_writer.add_summary(summary_str, step)

        # Optimizing the G_B network
        _, fake_A_temp, summary_str = self.session.run([self.g_B_trainer, self.fake_images_a, self.g_B_loss_summ],
                                                       feed_dict={
                                                           self.input_a: batch['images_a'],
                                                           self.input_b: batch['images_b'],
                                                           self.learning_rate: curr_lr
                                                       })
        self.summary_writer.add_summary(summary_str, step)

        fake_A_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_A_temp, self.fake_images_A)

        # Optimizing the D_A network
        _, summary_str = self.session.run(
            [self.d_A_trainer, self.d_A_loss_summ],
            feed_dict={
                self.input_a: batch['images_a'],
                self.input_b: batch['images_b'],
                self.learning_rate: curr_lr,
                self.fake_pool_A: fake_A_temp1
            })
        self.summary_writer.add_summary(summary_str, step)

        self.summary_writer.flush()
        self.num_fake_inputs += 1

    # Generate images from test noise
    def test_eval(self, image_a, image_b, step):
        summary = self.session.run(self.gen_image_summary_op, feed_dict={self.input_a: image_a, self.input_b: image_b})
        self.summary_writer.add_summary(summary, step)

    def run(self):
        BATCH_SIZE = 1
        base_lr = 0.0002

        # Iterate through epochs
        for epoch in range(FLAGS.model.optimization.epochs):
            print("Epoch %d" % epoch, flush=True)
            if epoch < 100:
                curr_lr = base_lr
            else:
                curr_lr = base_lr - base_lr * (epoch - 100) / 100

            # Read it every time to not skip some healthy images
            # Due to shuffling method.
            healthy, cancer = read_small(FLAGS.data.in_file)
            cancer, healthy = shuffle(cancer, healthy)
            epoch_steps = min(healthy.shape[0], cancer.shape[0])
            for n_batch, (batch_a, batch_b) in enumerate(
                    zip(next_batch(healthy, BATCH_SIZE), next_batch(cancer, BATCH_SIZE))):
                if batch_a.shape[0] == 0 or batch_b.shape[0] == 0:
                    break
                step = epoch * epoch_steps + n_batch

                inputs = {"images_a": batch_a, "images_b": batch_b}
                self.train_batch(inputs, step, curr_lr)

                # Test noise
                if n_batch % FLAGS.training.log_interval == 0:
                    self.test_eval(batch_a, batch_b, step)
                if n_batch % 1000 == 0:
                    self.saver.save(self.session, os.path.join(self.logdir, FLAGS.training.checkpoint_dir,
                                                               "model.ckpt"))

            # End of epoch
            if epoch % FLAGS.training.save_interval == 0:
                self.saver.save(self.session, os.path.join(self.logdir, FLAGS.training.checkpoint_dir, "model.ckpt"))


def run():
    CycleGan().run()
