# _*_ coding: utf-8 _*_

"""
A simple TensorFlow example of Linear Regression.

Author: Genpeng Xu
Date:	2019/04/09
"""

import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt

# Own scaffolds
from util.data_util import load_birth_life_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LOG_DIR = './runs'


def huber_loss(y_true, y_pred, delta=14.0):
    residual = tf.abs(y_true - y_pred)

    def f1():
        return 0.5 * tf.square(residual)

    def f2():
        return delta * residual - 0.5 * tf.square(delta)

    return tf.cond(residual > delta, f2, f1)


def main():
    data_file = "../data/birth_life_2010.txt"
    data = load_birth_life_data(data_file)
    n_samples = len(data)
    run_label = "uni_var_linreg_placeholder"

    # training parameters
    n_epochs = 100

    # create summary file
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    run_dir = os.path.join(LOG_DIR, run_label)
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    summary_dir = os.path.join(run_dir, 'summaries')
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    train_summary_dir = os.path.join(summary_dir, 'train')
    test_summary_dir = os.path.join(summary_dir, 'test')
    if not os.path.exists(train_summary_dir):
        os.mkdir(train_summary_dir)
    if not os.path.exists(test_summary_dir):
        os.mkdir(test_summary_dir)

    # Assemble a graph
    # ===================================================================================== #

    x = tf.placeholder(dtype=tf.float32, name='x')
    y = tf.placeholder(dtype=tf.float32, name='y')

    w = tf.get_variable(name='w', dtype=tf.float32, initializer=tf.constant(0.1))
    b = tf.get_variable(name='b', dtype=tf.float32, initializer=tf.constant(0.1))

    with tf.name_scope('y_'):
        y_ = w * x + b

    # loss
    with tf.name_scope('loss'):
        loss = huber_loss(y, y_)

    with tf.name_scope('train_op'):
        train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)

    init = tf.global_variables_initializer()

    # Use a session to execute operations in the graph
    # ===================================================================================== #

    print("[INFO] Starting training...")
    t0 = time.time()

    with tf.Session() as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        for i in range(1, n_epochs + 1):
            total_loss = 0
            for xx, yy in data:
                loss_train, _ = sess.run(fetches=[loss, train_op],
                                         feed_dict={x: xx, y: yy})
                total_loss += loss_train
            print("[Train] epoch: %03d, loss: %f" % (i, total_loss / n_samples))
        w_out, b_out = w.eval(), b.eval()

        train_writer.close()

    print("[INFO] Training finished! ( ^ _ ^ ) V")
    print("[INFO] Done in %f seconds." % (time.time() - t0))

    plt.plot(data[:, 0], data[:, 1], 'bo', label='Real data')
    plt.plot(data[:, 0], data[:, 0] * w_out + b_out, 'r', label='Predicted data')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
