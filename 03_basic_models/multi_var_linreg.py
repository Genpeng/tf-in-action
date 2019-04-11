# _*_ coding: utf-8 _*_

"""
A Linear regression example using TensorFlow.

TODO: The model can not normally train.

Author: Genpeng Xu
Date:   2019/03/27
"""

import os
import tensorflow as tf
from dataset.boston import BostonPriceData

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LOG_DIR = './runs'


def main():
    batch_size = 200
    train_steps = 10000
    train_data, test_data = BostonPriceData(), None
    run_label = "multi_var_linreg"

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
    # ============================================================================== #

    X = tf.placeholder(dtype=tf.float32, shape=[None, 13], name='X')  # (None, 13)
    y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')  # (None, )

    w = tf.get_variable(name='w', shape=[X.get_shape()[-1], 1],
                        initializer=tf.truncated_normal_initializer(stddev=0.02))  # (13, 1)
    b = tf.get_variable(name='b', dtype=tf.float32, initializer=tf.constant(0.0))

    # output
    with tf.name_scope('y_'):
        y_ = tf.matmul(X, w) + b  # (None, 1)

    # loss
    with tf.name_scope('loss'):
        y_reshaped = tf.reshape(y, shape=(-1, 1))  # (None, 1)
        loss = tf.reduce_mean(tf.square(y_ - y_reshaped))

    with tf.name_scope('train_op'):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    init = tf.global_variables_initializer()

    # Use a session to execute the graph
    # ============================================================================== #

    with tf.Session() as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        batch_data_train, batch_targets_train = train_data.next_batch(batch_size=batch_size)
        print(sess.run([y_, loss], feed_dict={X: batch_data_train, y: batch_targets_train}))

        for i in range(1, train_steps + 1):
            batch_data_train, batch_targets_train = train_data.next_batch(batch_size=batch_size)
            loss_train, _ = sess.run(fetches=[loss, train_op],
                                     feed_dict={X: batch_data_train, y: batch_targets_train})
            if i % 10 == 0:
                print("[Train] Step: %4d, loss: %4.5f" % (i, loss_train))
            if i % 50 == 0:
                test_data = BostonPriceData(need_shuffle=False)
                batch_data_test, batch_targets_test = test_data.next_batch(506)
                loss_test = sess.run(fetches=loss,
                                     feed_dict={X: batch_data_test, y: batch_targets_test})
                print()
                print("[Test ] Step: %4d, loss: %4.5f" % (i, loss_test))
                print()

        train_writer.close()


if __name__ == '__main__':
    main()
