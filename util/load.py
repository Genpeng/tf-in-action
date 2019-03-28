# _*_ coding: utf-8 _*_

"""
Some utility functions about loading data.

Author: Genpeng Xu (xgp1227@gmail.com)
Data:   2019/03/25
"""

import pickle
import numpy as np


def load_cifar_data(filepath):
    """Load CIFAR-10 dataset from file and return samples and its corresponding labels."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']


def load_birth_life_data(filepath):
    """Load World Development Indicators dataset."""
    with open(filepath, 'r') as f:
        text = f.readlines()[1:]
    data = [line[:-1].split('\t') for line in text]
    births = [float(line[1]) for line in data]
    lifes = [float(line[2]) for line in data]
    return np.asarray(list(zip(births, lifes)), dtype=np.float32)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Generate a batch iterator for dataset."""
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = (data_size - 1) // batch_size + 1
    for epoch in range(num_epochs):
        if shuffle:
            indices_shuffled = np.random.permutation(data_size)
            data_shuffled = data[indices_shuffled]
        else:
            data_shuffled = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield data_shuffled[start_index:end_index]


if __name__ == '__main__':
    # 测试 load_cifar_data
    # import os
    # cifar_dir = "../data/cifar-10-batches-py"
    # data, labels = load_cifar_data(os.path.join(cifar_dir, 'data_batch_1'))
    # print(data[:2])
    # print(labels[:2])

    # 测试 load_birth_life_data
    # birth_life_dir = "../data/birth_life_2010.txt"
    # data = load_birth_life_data(birth_life_dir)
    # print(data.shape)
    # print(data[:5])

    # 测试 batch_iter
    X = np.ones((50, 3))
    y = np.arange(50)
    data = np.array(list(zip(X, y)))
    batches = batch_iter(data, batch_size=20, num_epochs=1)
    for batch in batches:
        X_batch, y_batch = zip(*batch)
        print(X_batch)
        print(y_batch)

