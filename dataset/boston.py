# _*_ coding: utf-8 _*_

"""
A utility class for loading boston house-prices dataset.

Author: Genpeng Xu
Date:   2019/03/27
"""

import numpy as np
from sklearn.datasets import load_boston


class BostonPriceData:
    """Boston house-prices dataset."""

    def __init__(self, need_shuffle=True):
        boston_data = load_boston()
        self._data = boston_data.data
        self._targets = boston_data.target
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        indices_shuffled = np.random.permutation(self._num_examples)
        self._data = self._data[indices_shuffled]
        self._targets = self._targets[indices_shuffled]

    def next_batch(self, batch_size=20):
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("There have no more data!!!")
        if end_indicator > self._num_examples:
            raise Exception("The size of one batch is larger than the number of examples!!!")
        batch_data = self._data[self._indicator:end_indicator]
        batch_targets = self._targets[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_targets


def main():
    boston_data = load_boston()
    data, targets = boston_data.data, boston_data.target
    print(data.shape)
    print(data[:5])
    print(targets.shape)
    print(targets[:5])

    print()
    print()

    boston_data = BostonPriceData(False)
    batch_data, batch_targets = boston_data.next_batch(5)
    print(batch_data.shape)
    print(batch_data)
    print(batch_targets.shape)
    print(batch_targets)


if __name__ == '__main__':
    main()
