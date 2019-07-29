import os
import sys

import tensorflow as tf

import ops2

sys.path.append(os.getcwd())


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class NPCS_MLP:
    def __init__(self, config):
        self.X, self.y, self.X_val, self.y_val, self.d_dim = ops2.get_data(config.data, config.fill_points, 1.0, config)
        self.config = config
        self.limit = 1.0
        self.ev = None

    def positive(self, l):
        l = tf.assign(l, self.limit)
        return l

    def neg(self, l, delta_l):
        l = tf.assign(l, l + delta_l)
        l = tf.cond(l >= 1.0, true_fn=lambda: self.positive(l), false_fn=lambda: l)
        return l

    def update_l(self, l, delta_l):
        l = tf.cond(l >= 1.0, true_fn=lambda: self.positive(l), false_fn=lambda: self.neg(l, delta_l))
        return l

    # def mlp6(self, x, y, l):
    #     with tf.variable_scope('mlp'):
    #         layer = ops2.activation(self.config.use_act, tf.layers.dense(x, 200, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer1'), l)
    #         layer = tf.layers.batch_normalization(layer, training=True)
    #         layer1 = layer
    #         layer = ops2.activation(self.config.use_act, tf.layers.dense(layer1, 200, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer2'), l)
    #         layer = tf.layers.batch_normalization(layer, training=True)
    #         layer2 = layer + layer1
    #         layer = ops2.activation(self.config.use_act, tf.layers.dense(layer2,200, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer3'), l)
    #         layer = tf.layers.batch_normalization(layer, training=True)
    #         layer3 = layer + layer2
    #         layer = ops2.activation(self.config.use_act, tf.layers.dense(layer3, 200, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer4'), l)
    #         layer = tf.layers.batch_normalization(layer, training=True)
    #         layer4 = layer + layer3
    #         layer = ops2.activation(self.config.use_act, tf.layers.dense(layer4, 200, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer5'), l)
    #         layer = tf.layers.batch_normalization(layer, training=True)
    #         layer5 = layer + layer4
    #         pred = ops2.activation(self.config.use_act, tf.layers.dense(layer5, 1, name='layer6'), l)
    #         #pred =  tf.nn.sigmoid(tf.layers.dense(layer, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer6'))
    #         #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( labels=y, logits=pred, name='loss'))
    #         loss = tf.reduce_mean(tf.square(y - pred))
    #         return loss, pred

    def mlp6(self, x, y, l):
        with tf.variable_scope('mlp'):
            layer = ops2.activation(self.config.use_act, tf.layers.dense(x, 200,
                                                                         kernel_initializer=tf.glorot_uniform_initializer(),
                                                                         name='layer1'), l)
            layer = ops2.activation(self.config.use_act, tf.layers.dense(layer, 200,
                                                                         kernel_initializer=tf.glorot_uniform_initializer(),
                                                                         name='layer2'), l)
            layer = ops2.activation(self.config.use_act, tf.layers.dense(layer, 200,
                                                                         kernel_initializer=tf.glorot_uniform_initializer(),
                                                                         name='layer3'), l)
            layer = ops2.activation(self.config.use_act, tf.layers.dense(layer, 200,
                                                                         kernel_initializer=tf.glorot_uniform_initializer(),
                                                                         name='layer4'), l)
            layer = ops2.activation(self.config.use_act, tf.layers.dense(layer, 200,
                                                                         kernel_initializer=tf.glorot_uniform_initializer(),
                                                                         name='layer5'), l)

            pred = ops2.activation(self.config.use_act, tf.layers.dense(layer, 1, name='layer6'), l)
            # pred =  tf.nn.sigmoid(tf.layers.dense(layer, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer6'))
            # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( labels=y, logits=pred, name='loss'))
            loss = tf.reduce_mean(tf.square(y - pred))
            return loss, pred

    def master_graph(self):
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, shape=[None, self.d_dim], name='x')
        y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
        l = tf.Variable(self.config.l_init, dtype=tf.float32, trainable=False, name='lamda')

        delta_l = tf.placeholder(dtype=tf.float32, shape=[], name='delta_l')
        l_prev = tf.placeholder(dtype=tf.float32, shape=[], name='lamda_prev')
        omega = tf.placeholder(dtype=tf.float32, shape=[], name='omega')
        lnorm = tf.placeholder(dtype=tf.float32, shape=[], name='lnorm')

        with tf.variable_scope('current'):
            loss_c, output_c = self.mlp6(x, y, l)
        with tf.variable_scope('prev'):
            loss_p, output_p = self.mlp6(x, y, l)
        with tf.variable_scope('prev2'):
            loss_p2, output_p2 = self.mlp6(x, y, l)

        if self.config.opt == 'adam':
            optimizer = tf.train.AdamOptimizer(self.config.lr)
        elif self.config.opt == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(self.config.lr)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.config.lr)

        grads_and_vars = optimizer.compute_gradients(loss_c)
        opt = optimizer.apply_gradients(grads_and_vars)
        grads, _ = list(zip(*grads_and_vars))
        norms = tf.global_norm(grads)

        # lambda update NPC
        l_new = self.update_l(l, delta_l)

        # secant update for lambda NPCS
        A = tf.trainable_variables(scope='current/network')
        B = tf.trainable_variables(scope='prev/network')
        C = tf.trainable_variables(scope='prev2/network')

        copy_op = ops2.copy_g(A, B)
        copy_op1 = ops2.copy_g(A, C)
        copy_op2 = ops2.copy_g(C, B)

        diff_op = ops2.diff_l(A, B, self.config)
        secant_op = ops2.secant_l(A, B, self.config)

        return AttrDict(locals())
