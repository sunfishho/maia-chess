#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import os
import random
import tensorflow as tf
import time
import bisect
from .lc0_az_policy_map import make_map
from .proto.net_pb2 import NetworkFormat

from .net import Net

from ..utils import printWithDate
import wandb
import pdb

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class ApplySqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ApplySqueezeExcitation, self).__init__(**kwargs)

    def build(self, input_dimens):
        self.reshape_size = input_dimens[1][1]

    def call(self, inputs):
        x = inputs[0]
        excited = inputs[1]
        gammas, betas = tf.split(tf.reshape(excited, [-1, self.reshape_size, 1, 1]), 2, axis=1)
        return tf.nn.sigmoid(gammas) * x + betas

class TFProcessDiscriminator:
    def __init__(self, cfg, name, collection_name):
        self.cfg = cfg
        self.name = name
        self.collection_name = collection_name
        self.net = Net()
        self.root_dir = os.path.join('models', self.collection_name, self.name)

        # Network structure
        self.RESIDUAL_FILTERS = self.cfg['disc_model']['filters']
        self.RESIDUAL_BLOCKS = self.cfg['disc_model']['residual_blocks']
        self.SE_ratio = self.cfg['disc_model']['se_ratio']
        self.FINAL_FLAT_BLOCKS = self.cfg['disc_model']['final_flat_blocks']
        self.FINAL_FLAT_HLS = self.cfg['disc_model']['final_flat_hls']

        self.policy_channels = self.cfg['disc_model'].get('policy_channels', 32)
        precision = self.cfg['disc_training'].get('precision', 'single')
        loss_scale = self.cfg['disc_training'].get('loss_scale', 128)

        if precision == 'single':
            self.model_dtype = tf.float32
        elif precision == 'half':
            self.model_dtype = tf.float16
        else:
            raise ValueError("Unknown precision: {}".format(precision))

        # Scale the loss to prevent gradient underflow
        self.loss_scale = 1 if self.model_dtype == tf.float32 else loss_scale

        policy_head = self.cfg['disc_model'].get('policy', 'convolution')
        value_head  = self.cfg['disc_model'].get('value', 'wdl')

        self.POLICY_HEAD = None
        self.VALUE_HEAD = None

        if policy_head == "classical":
            self.POLICY_HEAD = NetworkFormat.POLICY_CLASSICAL
        elif policy_head == "convolution":
            self.POLICY_HEAD = NetworkFormat.POLICY_CONVOLUTION
        else:
            raise ValueError(
                "Unknown policy head format: {}".format(policy_head))

        self.net.set_policyformat(self.POLICY_HEAD)

        if value_head == "classical":
            self.VALUE_HEAD = NetworkFormat.VALUE_CLASSICAL
            self.wdl = False
        elif value_head == "wdl":
            self.VALUE_HEAD = NetworkFormat.VALUE_WDL
            self.wdl = True
        else:
            raise ValueError(
                "Unknown value head format: {}".format(value_head))

        self.net.set_valueformat(self.VALUE_HEAD)

        self.swa_enabled = self.cfg['disc_training'].get('swa', False)

        # Limit momentum of SWA exponential average to 1 - 1/(swa_max_n + 1)
        self.swa_max_n = self.cfg['disc_training'].get('swa_max_n', 0)

        self.renorm_enabled = self.cfg['disc_training'].get('renorm', False)
        self.renorm_max_r = self.cfg['disc_training'].get('renorm_max_r', 1)
        self.renorm_max_d = self.cfg['disc_training'].get('renorm_max_d', 0)
        self.renorm_momentum = self.cfg['disc_training'].get('renorm_momentum', 0.99)

        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[self.cfg['gpu']], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[self.cfg['gpu']], True)
        if self.model_dtype == tf.float16:
            tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

        self.global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)

    # def init_v2(self, train_dataset, test_dataset):
    def init_v2(self, train_iter, test_iter):
        # self.train_dataset = train_dataset
        # self.test_dataset = test_dataset
        # self.train_iter = iter(train_dataset)
        # self.test_iter = iter(test_dataset)
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.init_net_v2()

    def init_net_v2(self):
        self.l2reg = tf.keras.regularizers.l2(l=0.5 * (0.0001))
        input_var = tf.keras.Input(shape=(112, 8*8))
        input_labels = tf.keras.Input(shape=1858, dtype=self.model_dtype)
        x_planes = tf.keras.layers.Reshape([112, 8, 8])(input_var)
        self.model = tf.keras.Model(inputs=(input_var, input_labels), outputs=self.construct_net_v2((x_planes, input_labels)))
        # swa_count initialized reguardless to make checkpoint code simpler.
        self.swa_count = tf.Variable(0., name='swa_count', trainable=False)
        self.swa_weights = None
        if self.swa_enabled:
            # Count of networks accumulated into SWA
            self.swa_weights = [tf.Variable(w, trainable=False) for w in self.model.weights]

        self.active_lr = 0.01
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lambda: self.active_lr,
                momentum=0.9, nesterov=True)
        self.orig_optimizer = self.optimizer
        if self.loss_scale != 1:
            self.optimizer = tf.keras.mixed_precision.experimental.\
                    LossScaleOptimizer(self.optimizer, self.loss_scale)

        def correct_policy(target, output):
            output = tf.cast(output, tf.float32)
            # Calculate loss on policy head
            if self.cfg['disc_training'].get('mask_legal_moves'):
                # extract mask for legal moves from target policy
                move_is_legal = tf.greater_equal(target, 0)
                # replace logits of illegal moves with large negative value (so that it doesn't affect policy of legal moves) without gradient
                illegal_filler = tf.zeros_like(output) - 1.0e10
                output = tf.where(move_is_legal, output, illegal_filler)
            # y_ still has -1 on illegal moves, flush them to 0
            target = tf.nn.relu(target)
            return target, output
        def policy_loss(target, output):
            target, output = correct_policy(target, output)
            policy_cross_entropy = \
                tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(target),
                                                        logits=output)

            return tf.reduce_mean(input_tensor=policy_cross_entropy)
        self.policy_loss_fn = policy_loss
        def policy_accuracy(target, output):
            target, output = correct_policy(target, output)
            return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(input=target, axis=1), tf.argmax(input=output, axis=1)), tf.float32))
        self.policy_accuracy_fn = policy_accuracy


        q_ratio = self.cfg['disc_training'].get('q_ratio', 0)
        assert 0 <= q_ratio <= 1

        # Linear conversion to scalar to compute MSE with, for comparison to old values
        wdl = tf.expand_dims(tf.constant([1.0, 0.0, -1.0]), 1)

        self.qMix = lambda z, q: q * q_ratio + z *(1 - q_ratio)
        # Loss on value head
        if self.wdl:
            def value_loss(target, output):
                output = tf.cast(output, tf.float32)
                value_cross_entropy = \
                    tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(target),
                                                    logits=output)
                return tf.reduce_mean(input_tensor=value_cross_entropy)
            self.value_loss_fn = value_loss
            def mse_loss(target, output):
                output = tf.cast(output, tf.float32)
                scalar_z_conv = tf.matmul(tf.nn.softmax(output), wdl)
                scalar_target = tf.matmul(target, wdl)
                return tf.reduce_mean(input_tensor=tf.math.squared_difference(scalar_target, scalar_z_conv))
            self.mse_loss_fn = mse_loss
        else:
            def value_loss(target, output):
                return tf.constant(0)
            self.value_loss_fn = value_loss
            def mse_loss(target, output):
                output = tf.cast(output, tf.float32)
                scalar_target = tf.matmul(target, wdl)
                return tf.reduce_mean(input_tensor=tf.math.squared_difference(scalar_target, output))
            self.mse_loss_fn = mse_loss

        pol_loss_w = self.cfg['disc_training']['policy_loss_weight']
        val_loss_w = self.cfg['disc_training']['value_loss_weight']
        self.lossMix = lambda policy, value: pol_loss_w * policy + val_loss_w * value

        def accuracy(target, output):
            output = tf.cast(output, tf.float32)
            return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(input=target, axis=1), tf.argmax(input=output, axis=1)), tf.float32))
        self.accuracy_fn = accuracy

        self.avg_policy_loss = []
        self.avg_value_loss = []
        self.avg_mse_loss = []
        self.avg_reg_term = []
        self.time_start = None
        self.last_steps = None
        # Set adaptive learning rate during training
        self.cfg['disc_training']['lr_boundaries'].sort()
        self.warmup_steps = self.cfg['disc_training'].get('warmup_steps', 0)
        self.lr = self.cfg['disc_training']['lr_values'][0]
        self.test_writer = tf.summary.create_file_writer(os.path.join(
                'runs',
                self.collection_name,
                self.name + '-test',
        ))
        self.train_writer = tf.summary.create_file_writer(os.path.join(
                'runs',
                self.collection_name,
                self.name + '-train',
        ))
        if self.swa_enabled:
            self.swa_writer = tf.summary.create_file_writer(os.path.join(
                    'runs',
                    self.collection_name,
                    self.name + '-swa-test',
            ))
        self.checkpoint = tf.train.Checkpoint(optimizer=self.orig_optimizer, model=self.model, global_step=self.global_step, swa_count=self.swa_count)
        self.checkpoint.listed = self.swa_weights
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.root_dir, max_to_keep=50, keep_checkpoint_every_n_hours=24)

    def replace_weights_v2(self, new_weights_orig):
        new_weights = [w for w in new_weights_orig]
        # self.model.weights ordering doesn't match up nicely, so first shuffle the new weights to match up.
        # input order is (for convolutional policy):
        # policy conv
        # policy bn * 4
        # policy raw conv and bias
        # value conv
        # value bn * 4
        # value dense with bias
        # value dense with bias
        #
        # output order is (for convolutional policy):
        # value conv
        # policy conv
        # value bn * 4
        # policy bn * 4
        # policy raw conv and bias
        # value dense with bias
        # value dense with bias
        new_weights[-5] = new_weights_orig[-10]
        new_weights[-6] = new_weights_orig[-11]
        new_weights[-7] = new_weights_orig[-12]
        new_weights[-8] = new_weights_orig[-13]
        new_weights[-9] = new_weights_orig[-14]
        new_weights[-10] = new_weights_orig[-15]
        new_weights[-11] = new_weights_orig[-5]
        new_weights[-12] = new_weights_orig[-6]
        new_weights[-13] = new_weights_orig[-7]
        new_weights[-14] = new_weights_orig[-8]
        new_weights[-15] = new_weights_orig[-16]
        new_weights[-16] = new_weights_orig[-9]

        all_evals = []
        offset = 0
        last_was_gamma = False
        for e, weights in enumerate(self.model.weights):
            source_idx = e+offset
            if weights.shape.ndims == 4:
                # Rescale rule50 related weights as clients do not normalize the input.
                if e == 0:
                    num_inputs = 112
                    # 50 move rule is the 110th input, or 109 starting from 0.
                    rule50_input = 109
                    for i in range(len(new_weights[source_idx])):
                        if (i % (num_inputs*9))//9 == rule50_input:
                            new_weights[source_idx][i] = new_weights[source_idx][i]*99

                # Convolution weights need a transpose
                #
                # TF (kYXInputOutput)
                # [filter_height, filter_width, in_channels, out_channels]
                #
                # Leela/cuDNN/Caffe (kOutputInputYX)
                # [output, input, filter_size, filter_size]
                s = weights.shape.as_list()
                shape = [s[i] for i in [3, 2, 0, 1]]
                new_weight = tf.constant(new_weights[source_idx], shape=shape)
                weights.assign(
                    tf.transpose(a=new_weight, perm=[2, 3, 1, 0]))
            elif weights.shape.ndims == 2:
                # Fully connected layers are [in, out] in TF
                #
                # [out, in] in Leela
                #
                s = weights.shape.as_list()
                shape = [s[i] for i in [1, 0]]
                new_weight = tf.constant(new_weights[source_idx], shape=shape)
                weights.assign(
                    tf.transpose(a=new_weight, perm=[1, 0]))
            else:
                # Can't populate renorm weights, but the current new_weight will need using elsewhere.
                if 'renorm' in weights.name:
                    offset-=1
                    continue
                # betas without gamms need to skip the gamma in the input.
                if 'beta:' in weights.name and not last_was_gamma:
                    source_idx+=1
                    offset+=1
                # Biases, batchnorm etc
                new_weight = tf.constant(new_weights[source_idx], shape=weights.shape)
                if 'stddev:' in weights.name:
                    weights.assign(tf.math.sqrt(new_weight + 1e-5))
                else:
                    weights.assign(new_weight)
                # need to use the variance to also populate the stddev for renorm, so adjust offset.
                if 'variance:' in weights.name and self.renorm_enabled:
                    offset-=1
            last_was_gamma = 'gamma:' in weights.name
        # Replace the SWA weights as well, ensuring swa accumulation is reset.
        if self.swa_enabled:
            self.swa_count.assign(tf.constant(0.))
            self.update_swa_v2()
        # This should result in identical file to the starting one
        # self.save_leelaz_weights_v2('restored.pb.gz')

    def restore_v2(self):
        if self.manager.latest_checkpoint is not None:
            print("Restoring from {0}".format(self.manager.latest_checkpoint))
            self.checkpoint.restore(self.manager.latest_checkpoint)

    def process_loop_v2(self, batch_size, test_batches, train_batches, batch_splits=1):
        # Get the initial steps value in case this is a resume from a step count
        # which is not a multiple of total_steps_per_cycle.
        steps = self.global_step.read_value()

        total_steps_per_cycle = self.cfg['disc_training']['total_steps_per_cycle']
        for _ in range(steps % total_steps_per_cycle, total_steps_per_cycle):
            self.process_v2(batch_size, test_batches, train_batches,
                    batch_splits=batch_splits)

    #@tf.function()
    def read_weights(self):
        return [w.read_value() for w in self.model.weights]

    #@tf.function()
    def process_inner_loop(self, x, y, z, q):
        with tf.GradientTape() as tape:
            labels_human = tf.tile(tf.constant([[0, 1]]), [x.shape[0], 1])
            labels_nonhuman = tf.tile(tf.constant([[1, 0]]), [x.shape[0], 1])

            non_human_move = self.gen_model(x, training=False)[0]
            non_human_move = tf.one_hot(tf.cast(tf.math.argmax(non_human_move,
                axis=1), tf.int32), y.shape[1], dtype=self.model_dtype)

            ynum = y.numpy()
            indices = ynum == -1
            ynum[indices] = 0
            y = tf.convert_to_tensor(ynum)

            human_policy = self.model((x, y), training=True)
            human_policy_loss = self.policy_loss_fn(labels_human, human_policy)

            nonhuman_policy  = self.model((x, non_human_move), training=True)
            nonhuman_policy_loss = self.policy_loss_fn(labels_nonhuman, nonhuman_policy)

            policy_loss = human_policy_loss + nonhuman_policy_loss

            reg_term = sum(self.model.losses)

            ## simpler version
            value_loss = 0.0
            mse_loss = 0.0
            total_loss = policy_loss

            ## TODO: add reg_term or not to total loss
            if self.loss_scale != 1:
                total_loss = self.optimizer.get_scaled_loss(total_loss)

        return policy_loss, value_loss, mse_loss, reg_term,\
                tape.gradient(total_loss, self.model.trainable_weights)

    def process_v2(self, batch_size, test_batches, train_batches, batch_splits=1):
        if not self.time_start:
            self.time_start = time.time()
        # Get the initial steps value before we do a training step.
        steps = self.global_step.read_value()
        if not self.last_steps:
            self.last_steps = steps

        if steps % self.cfg['disc_training']['test_steps'] == 0:
            self.calculate_summaries_v2(test_batches, steps)
            self.calculate_summaries_v2(10, steps, kind="Train")
            # self.calculate_summaries_v2(train_batches, steps, kind="Train")

        if self.swa_enabled:
            # split half of test_batches between testing regular weights and SWA weights
            test_batches //= 2

        # Make sure that ghost batch norm can be applied
        if batch_size % 64 != 0:
            # Adjust required batch size for batch splitting.
            required_factor = 64 * \
                self.cfg['disc_training'].get('num_batch_splits', 1)
            raise ValueError(
                'batch_size must be a multiple of {}'.format(required_factor))

        # Determine learning rate
        lr_values = self.cfg['disc_training']['lr_values']
        lr_boundaries = self.cfg['disc_training']['lr_boundaries']
        steps_total = steps % self.cfg['disc_training']['total_steps_per_cycle']
        self.lr = lr_values[bisect.bisect_right(lr_boundaries, steps_total)]
        if self.warmup_steps > 0 and steps < self.warmup_steps:
            self.lr = self.lr * tf.cast(steps + 1, tf.float32) / self.warmup_steps

        # Run training for this batch
        grads = None
        counter = 0

        for _ in range(batch_splits):
            counter += 1
            x, y, z, q = next(self.train_iter)
            policy_loss, value_loss, mse_loss, reg_term, new_grads = self.process_inner_loop(x, y, z, q)

            if not grads:
                grads = new_grads
            else:
                grads = [tf.math.add(a, b) for (a, b) in zip(grads, new_grads)]

            # Keep running averages
            # Google's paper scales MSE by 1/4 to a [0, 1] range, so do the same to
            # get comparable values.
            mse_loss /= 4.0

            self.avg_policy_loss.append(policy_loss)
            # if self.wdl:
                # self.avg_value_loss.append(value_loss)
            # self.avg_mse_loss.append(mse_loss)
            self.avg_reg_term.append(reg_term)

        # Gradients of batch splits are summed, not averaged like usual, so
        # need to scale lr accordingly to correct for this.
        self.active_lr = self.lr / batch_splits
        if self.loss_scale != 1:
            grads = self.optimizer.get_unscaled_gradients(grads)
        max_grad_norm = self.cfg['disc_training'].get('max_grad_norm', 10000.0) * batch_splits
        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # Update steps.
        self.global_step.assign_add(1)
        steps = self.global_step.read_value()

        if steps % self.cfg['disc_training']['train_avg_report_steps'] == 0:
            pol_loss_w = self.cfg['disc_training']['policy_loss_weight']
            val_loss_w = self.cfg['disc_training']['value_loss_weight']
            time_end = time.time()
            speed = 0
            if self.time_start:
                elapsed = time_end - self.time_start
                steps_elapsed = steps - self.last_steps
                speed = batch_size * (tf.cast(steps_elapsed, tf.float32) / elapsed)

            avg_policy_loss = np.mean(self.avg_policy_loss or [0])
            avg_reg_term = np.mean(self.avg_reg_term or [0])

            printWithDate("D: step {}, lr={:g} policy={:g} ({:g} pos/s)".format(steps,
                self.lr, avg_policy_loss,
                speed))
            self.time_start = time_end
            self.last_steps = steps

            wandb.log({"Discriminator LR": self.lr , "step":steps})
            wandb.log({"Discriminator Reg term": avg_reg_term , "step":steps})
            wandb.log({"Avg Discriminator Train Policy Loss": avg_policy_loss , "step":steps})

            self.avg_policy_loss = []
            self.avg_reg_term = []

        if self.swa_enabled and steps % self.cfg['disc_training']['swa_steps'] == 0:
            assert False
            self.update_swa_v2()


        if False:
            self.manager.save()
            print("Model saved in file: {}".format(self.manager.latest_checkpoint))
            evaled_steps = steps.numpy()
            leela_path = self.manager.latest_checkpoint + "-" + str(evaled_steps)
            swa_path = self.manager.latest_checkpoint + "-swa-" + str(evaled_steps)
            self.net.pb.training_params.training_steps = evaled_steps
            self.save_leelaz_weights_v2(leela_path)
            print("Weights saved in file: {}".format(leela_path))
            if self.swa_enabled:
                self.save_swa_weights_v2(swa_path)
                print("SWA Weights saved in file: {}".format(swa_path))

    def calculate_swa_summaries_v2(self, test_batches, steps):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        true_test_writer, self.test_writer = self.test_writer, self.swa_writer
        print('swa', end=' ')
        self.calculate_summaries_v2(test_batches, steps)
        self.test_writer = true_test_writer
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    @tf.function()
    def calculate_test_summaries_inner_loop(self, x, y, z, q):
        labels_human = tf.tile(tf.constant([[0, 1]]), [x.shape[0], 1])
        labels_nonhuman = tf.tile(tf.constant([[1, 0]]), [x.shape[0], 1])

        non_human_move = self.gen_model(x, training=False)[0]
        # non_human_move = tf.math.argmax(non_human_move[0], axis=1)
        # TODO: convert into zeros or ones
        non_human_move = tf.one_hot(tf.cast(tf.math.argmax(non_human_move,
            axis=1), tf.int32), y.shape[1], dtype=self.model_dtype)

        human_policy = self.model((x, y), training=True)
        human_policy_accuracy = self.policy_accuracy_fn(labels_human, human_policy)
        human_policy_loss = self.policy_loss_fn(labels_human, human_policy)

        nonhuman_policy = self.model((x, non_human_move), training=True)
        nonhuman_policy_accuracy = self.policy_accuracy_fn(labels_nonhuman, nonhuman_policy)
        nonhuman_policy_loss = self.policy_loss_fn(labels_nonhuman, nonhuman_policy)

        policy_loss = human_policy_loss + nonhuman_policy_loss

        policy_accuracy = (human_policy_accuracy + nonhuman_policy_accuracy) / 2

        value_loss = 0.0
        mse_loss = 0.0
        value_accuracy = 0.0

        return policy_loss, value_loss, mse_loss, policy_accuracy, value_accuracy

    def calculate_summaries_v2(self, batches, steps, kind='Test'):
        start = time.time()
        sum_policy_accuracy = 0
        sum_value_accuracy = 0
        sum_mse = 0
        sum_policy = 0
        sum_value = 0
        for i in range(0, batches):
            if kind == 'Train':
                x, y, z, q = next(self.train_iter)
            elif kind == 'Test':
                x, y, z, q = next(self.test_iter)
            policy_loss, _, _, policy_accuracy, _ = \
                    self.calculate_test_summaries_inner_loop(x, y, z, q)
            sum_policy_accuracy += policy_accuracy
            # sum_mse += mse_loss
            sum_policy += policy_loss
            # if self.wdl:
                # sum_value_accuracy += value_accuracy
                # sum_value += value_loss

        sum_policy_accuracy /= batches
        sum_policy_accuracy *= 100
        sum_policy /= batches
        sum_value /= batches
        if self.wdl:
            sum_value_accuracy /= batches
            sum_value_accuracy *= 100
        # Additionally rescale to [0, 1] so divide by 4
        sum_mse /= (4.0 * batches)
        self.net.pb.training_params.learning_rate = self.lr
        self.net.pb.training_params.mse_loss = sum_mse
        self.net.pb.training_params.policy_loss = sum_policy
        # TODO store value and value accuracy in pb
        self.net.pb.training_params.accuracy = sum_policy_accuracy

        printWithDate("D, {}: step {}, policy={:g} policy accuracy={:g}%".\
            format(kind, steps, sum_policy, sum_policy_accuracy))

        wandb.log({f"Discriminator {kind} Policy Loss": sum_policy , "step":steps})
        wandb.log({f"Discriminator {kind} Policy Accuracy": sum_policy_accuracy , "step":steps})

        print("calculate summaries took: ", time.time()-start)

    # @tf.function()
    def compute_update_ratio_v2(self, before_weights, after_weights, steps):
        """Compute the ratio of gradient norm to weight norm.

        Adapted from https://github.com/tensorflow/minigo/blob/c923cd5b11f7d417c9541ad61414bf175a84dc31/dual_net.py#L567
        """
        deltas = [after - before for after,
                  before in zip(after_weights, before_weights)]
        delta_norms = [tf.math.reduce_euclidean_norm(d) for d in deltas]
        weight_norms = [tf.math.reduce_euclidean_norm(w) for w in before_weights]
        ratios = [(tensor.name, tf.cond(w != 0., lambda: d / w, lambda: -1.)) for d, w, tensor in zip(delta_norms, weight_norms, self.model.weights) if not 'moving' in tensor.name]
        for name, ratio in ratios:
            tf.summary.scalar('update_ratios/' + name, ratio, step=steps)
        # Filtering is hard, so just push infinities/NaNs to an unreasonably large value.
        ratios = [tf.cond(r > 0, lambda: tf.math.log(r) / 2.30258509299, lambda: 200.) for (_, r) in ratios]
        tf.summary.histogram('update_ratios_log10', tf.stack(ratios), buckets=1000, step=steps)

    def update_swa_v2(self):
        num = self.swa_count.read_value()
        for (w, swa) in zip(self.model.weights, self.swa_weights):
            swa.assign(swa.read_value() * (num / (num + 1.)) + w.read_value() * (1. / (num + 1.)))
        self.swa_count.assign(min(num + 1., self.swa_max_n))

    def save_swa_weights_v2(self, filename):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        self.save_leelaz_weights_v2(filename)
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    def save_leelaz_weights_v2(self, filename):
        return 0
        # all_tensors = []
        # all_weights = []
        # last_was_gamma = False
        # for weights in self.model.weights:
        #     work_weights = None
        #     if weights.shape.ndims == 4:
        #         # Convolution weights need a transpose
        #         #
        #         # TF (kYXInputOutput)
        #         # [filter_height, filter_width, in_channels, out_channels]
        #         #
        #         # Leela/cuDNN/Caffe (kOutputInputYX)
        #         # [output, input, filter_size, filter_size]
        #         work_weights = tf.transpose(a=weights, perm=[3, 2, 0, 1])
        #     elif weights.shape.ndims == 2:
        #         # Fully connected layers are [in, out] in TF
        #         #
        #         # [out, in] in Leela
        #         #
        #         work_weights = tf.transpose(a=weights, perm=[1, 0])
        #     else:
        #         # batch renorm has extra weights, but we don't know what to do with them.
        #         if 'renorm' in weights.name:
        #             continue
        #         # renorm has variance, but it is not the primary source of truth
        #         if 'variance:' in weights.name and self.renorm_enabled:
        #             continue
        #         # Renorm has moving stddev not variance, undo the transform to make it compatible.
        #         if 'stddev:' in weights.name:
        #             all_tensors.append(tf.math.square(weights) - 1e-5)
        #             continue
        #         # Biases, batchnorm etc
        #         # pb expects every batch norm to have gammas, but not all of our
        #         # batch norms have gammas, so manually add pretend gammas.
        #         if 'beta:' in weights.name and not last_was_gamma:
        #             all_tensors.append(tf.ones_like(weights))
        #         work_weights = weights.read_value()
        #     all_tensors.append(work_weights)
        #     last_was_gamma = 'gamma:' in weights.name

        # # HACK: model weights ordering is some kind of breadth first traversal,
        # # but pb expects a specific ordering which BFT is not a match for once
        # # we get to the heads. Apply manual permutation.
        # # This is fragile and at minimum should have some checks to ensure it isn't breaking things.
        # #TODO: also support classic policy head as it has a different set of layers and hence changes the permutation.
        # permuted_tensors = [w for w in all_tensors]
        # permuted_tensors[-5] = all_tensors[-11]
        # permuted_tensors[-6] = all_tensors[-12]
        # permuted_tensors[-7] = all_tensors[-13]
        # permuted_tensors[-8] = all_tensors[-14]
        # permuted_tensors[-9] = all_tensors[-16]
        # permuted_tensors[-10] = all_tensors[-5]
        # permuted_tensors[-11] = all_tensors[-6]
        # permuted_tensors[-12] = all_tensors[-7]
        # permuted_tensors[-13] = all_tensors[-8]
        # permuted_tensors[-14] = all_tensors[-9]
        # permuted_tensors[-15] = all_tensors[-10]
        # permuted_tensors[-16] = all_tensors[-15]
        # all_tensors = permuted_tensors

        # for e, nparray in enumerate(all_tensors):
        #     # Rescale rule50 related weights as clients do not normalize the input.
        #     if e == 0:
        #         num_inputs = 112
        #         # 50 move rule is the 110th input, or 109 starting from 0.
        #         rule50_input = 109
        #         wt_flt = []
        #         for i, weight in enumerate(np.ravel(nparray)):
        #             if (i % (num_inputs*9))//9 == rule50_input:
        #                 wt_flt.append(weight/99)
        #             else:
        #                 wt_flt.append(weight)
        #     else:
        #         wt_flt = [wt for wt in np.ravel(nparray)]
        #     all_weights.append(wt_flt)

        # # self.net.fill_net(all_weights)
        # # self.net.save_proto(filename)

    def batch_norm_v2(self, input, scale=False):
        if self.renorm_enabled:
            clipping = {
                "rmin": 1.0/self.renorm_max_r,
                "rmax": self.renorm_max_r,
                "dmax": self.renorm_max_d
                }
            return tf.keras.layers.BatchNormalization(
                epsilon=1e-5, axis=1, fused=False, center=True,
                scale=scale, renorm=True, renorm_clipping=clipping,
                renorm_momentum=self.renorm_momentum)(input)
        else:
            return tf.keras.layers.BatchNormalization(
                epsilon=1e-5, axis=1, fused=False, center=True,
                scale=scale, virtual_batch_size=64)(input)

    def squeeze_excitation_v2(self, inputs, channels):
        assert channels % self.SE_ratio == 0

        pooled = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(inputs)
        squeezed = tf.keras.layers.Activation('relu')(tf.keras.layers.Dense(channels // self.SE_ratio, kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg)(pooled))
        excited = tf.keras.layers.Dense(2 * channels, kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg)(squeezed)
        return ApplySqueezeExcitation()([inputs, excited])

    def conv_block_v2(self, inputs, filter_size, output_channels, bn_scale=False):
        conv = tf.keras.layers.Conv2D(output_channels, filter_size, use_bias=False, padding='same', kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg, data_format='channels_first')(inputs)
        return tf.keras.layers.Activation('relu')(self.batch_norm_v2(conv, scale=bn_scale))

    def residual_block_v2(self, inputs, channels):
        # As shown in figure 10 of the maia paper: conv2d + relu -> conv2d -> fully connected block -> relu
        conv1 = tf.keras.layers.Conv2D(channels, 3, use_bias=False, padding='same', kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg, data_format='channels_first')(inputs)
        out1 = tf.keras.layers.Activation('relu')(self.batch_norm_v2(conv1, scale=False))
        conv2 = tf.keras.layers.Conv2D(channels, 3, use_bias=False, padding='same', kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg, data_format='channels_first')(out1)
        out2 = self.squeeze_excitation_v2(self.batch_norm_v2(conv2, scale=True), channels)
        return tf.keras.layers.Activation('relu')(tf.keras.layers.add([inputs, out2]))

    def construct_net_v2(self, inputs):
        move = inputs[1]
        inputs = inputs[0]
        flow = self.conv_block_v2(inputs, filter_size=3, output_channels=self.RESIDUAL_FILTERS, bn_scale=True)

        for _ in range(0, self.RESIDUAL_BLOCKS):
            flow = self.residual_block_v2(flow, self.RESIDUAL_FILTERS)

        # Policy head
        if self.POLICY_HEAD == NetworkFormat.POLICY_CONVOLUTION:
            conv_pol = self.conv_block_v2(flow, filter_size=3,
                    output_channels=self.RESIDUAL_FILTERS)
            conv_pol2 = tf.keras.layers.Conv2D(80, 3, use_bias=True, padding='same',
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=self.l2reg,
                    bias_regularizer=self.l2reg, data_format='channels_first')(conv_pol)
            conv_pol2_flat = tf.reshape(conv_pol2, [-1, 80*8*8])
            final_flat = tf.concat([conv_pol2_flat, move], axis=1)

            for _ in range(0, self.FINAL_FLAT_BLOCKS):
                final_flat = tf.keras.layers.Dense(self.FINAL_FLAT_HLS,
                        activation='relu',
                        kernel_initializer='glorot_normal',
                        kernel_regularizer=self.l2reg,
                        bias_regularizer=self.l2reg)(final_flat)

            h_fc1 = tf.keras.layers.Dense(2,
                    activation='softmax',
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=self.l2reg,
                    bias_regularizer=self.l2reg)(final_flat)
        else:
            raise ValueError(
                "Unknown policy head type {}".format(self.POLICY_HEAD))

        return h_fc1
