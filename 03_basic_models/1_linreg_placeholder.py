# _*_ coding: utf-8 _*_

"""
A Linear regression example using TensorFlow.

Author: Genpeng Xu
Date:   2019/03/27
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
from util.load import load_birth_life_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def huber_loss(labels, predictions, delta=14.0):
    residual = tf.abs(labels - predictions)

    def f1():
        return 0.5 * tf.square(residual)

    def f2():
        return delta * residual - 0.5 * tf.square(delta)

    return tf.cond(residual < delta, f1, f2)


def main():
    birth_life_data_dir = "../data/birth_life_2010.txt"
    data = load_birth_life_data(birth_life_data_dir)

    num_epochs = 100

    # Assemble a graph
    # =============================================================================== #

    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')

    w = tf.get_variable(name='w', initializer=tf.constant(0.0))
    b = tf.get_variable(name='b', initializer=tf.constant(0.0))

    # output
    with tf.name_scope('output'):
        y_ = x * w + b

    # loss
    with tf.name_scope('loss'):
        loss = huber_loss(labels=y, predictions=y_)

    with tf.name_scope('train_op'):
        train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)

    # Use a session to execute the graph
    # =============================================================================== #

    print("[INFO] Start training...")
    t0 = time()

    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter("./graph/1_linreg_placeholder/", tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1, num_epochs + 1):
            total_loss = 0
            for x_, y_ in data:
                l, _ = sess.run([loss, train_op], feed_dict={x: x_, y: y_})
                total_loss += l
            print("Epoch: %d, loss: %4.5f" % (i, total_loss / data.shape[0]))
        w_out, b_out = sess.run([w, b])
    writer.close()

    print("[INFO] Done in %f seconds." % (time() - t0))
    print("[INFO] Training finished! ( ^ _ ^ ) V")

    # Plot the results
    # =============================================================================== #

    plt.plot(data[:, 0], data[:, 1], 'bo', label='Real data')
    plt.plot(data[:, 0], data[:, 0] * w_out + b_out, 'r', label='Predicted data')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
