#!/bin/env python3
# SBATCH -N 1      # nodes requested
# SBATCH -n 1      # tasks requested
# SBATCH -c 2      # cores requested# you want to use gpu or cpu
# SBATCH --gres=gpu:2
# SBATCH -t 04:01:00  # time requested in hour:minute:second


import os
import sys

sys.path.append(os.getcwd())

import matplotlib

matplotlib.use('agg')

import numpy as np
import tensorflow as tf
from utils.config import process_config
from mlp import *
import ops2


# from utils.dirs import create_dirs
# from utils.logger import Logger
# from utils.utils import get_args


def main():
    losses = []
    lamdas = []
    norms = []
    val_losses = []
    # args = get_args()
    config = process_config('example.json')
    val_loss = 0.1
    npcs_mlp = NPCS_MLP(config)
    step = 0
    l_dash = config.l_init
    graph = npcs_mlp.master_graph()
    np.random.seed(config.seed)
    saver = tf.train.Saver()
    tf.set_random_seed(config.seed)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(config.max_steps):
            step = step + 1
            if step == config.max_steps:
                break;

            if (step % config.val_freq == 0):
                val_loss = sess.run(graph.loss_c, feed_dict={graph.x: npcs_mlp.X_val, graph.y: npcs_mlp.y_val})
                val_losses.append(val_loss)
            else:
                val_losses.append(val_loss)

            x_b, y_b = ops2.get_batch_with_labels(config.batch_size, npcs_mlp.X, npcs_mlp.y)

            norm, _ = sess.run([graph.norms, graph.opt],
                               feed_dict={graph.x: x_b, graph.y: y_b})
            norms.append(norm)

            loss = sess.run(graph.loss_c, feed_dict={graph.x: x_b, graph.y: y_b})  # No name collision anymore.
            losses.append(loss)
            # print(loss)
            lamdas.append(sess.run(graph.l))
            assert not np.isnan(loss)

            if (step % config.u_freq == 0) and (step > 0) and (l_dash < 1.0) and (
                    config.secant_method) and (step > config.adaptive_start):  # stop the secant after 0.99
                l_dash = sess.run(graph.l_new, feed_dict={graph.delta_l: config.delta_l})
                config.delta_l = ops2.adaptive_lambda(config, step, norms)

        # print("Step: %d, lambda %g, Loss: %g" %(step, sess.run(graph.l), loss) )
        # code = sess.run(graph.code, feed_dict={graph.x: npcs_ae.X})
        if not os.path.exists('./results/' + str(config.omega_exp)):
            os.makedirs('./results/' + str(config.omega_exp))
        saver.save(sess, './results/' + str(config.omega_exp) + '/model/ds')
        np.save('./results/' + str(config.omega_exp) + '/losses.npy', losses)
        np.save('./results/' + str(config.omega_exp) + '/lamdas.npy', lamdas)
        np.save('./results/' + str(config.omega_exp) + '/norms.npy', norms)
        np.save('./results/' + str(config.omega_exp) + '/val_losses.npy', val_losses)
        # ops2.save_plots(code, losses, val_losses, lamdas, norms, config)


if __name__ == '__main__':
    main()
