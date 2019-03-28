# _*_ coding: utf-8 _*_

"""
A Linear regression example using TensorFlow.

Author: Genpeng Xu
Date:   2019/03/27
"""

import os
import tensorflow as tf
from dataset.boston import BostonPriceData

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    batch_size = 20
    train_steps = 1000
    train_data, test_data = BostonPriceData(), None

    # Assemble a graph
    # ============================================================================== #

    X = tf.placeholder(dtype=tf.float32, shape=[None, 13], name='X')  # (None, 13)
    y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')  # (None, )

    w = tf.get_variable(name='w', shape=[X.get_shape()[-1], 1],
                        initializer=tf.random_normal_initializer(0, 1))  # (13, 1)
    b = tf.get_variable(name='b', shape=[1], initializer=tf.constant_initializer(0.0))  # (1, )

    # output
    with tf.name_scope('output'):
        y_ = tf.add(tf.matmul(X, w), b)  # (None, 1)

    # loss
    with tf.name_scope('loss'):
        y_reshaped = tf.reshape(y, shape=(-1, 1))
        loss = tf.reduce_mean(tf.square(y_ - y_reshaped), name='loss')

    with tf.name_scope('train_op'):
        train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)

    # Use a session to execute the graph
    # ============================================================================== #

    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter("./graph/linear_regression_placeholder/", tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(init)

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

    writer.close()


if __name__ == '__main__':
    main()
