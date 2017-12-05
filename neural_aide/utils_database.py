#!/usr/bin/python
# coding: utf-8

import logging
import numpy as np


def normalize_npy_database(data, axis=0):
    """
    For each column, remove mean and divide by standard deviation.
    Params:
        data (np.array): Data to be normalized.
        axis (integer): axis along which data will be normalized.

    Return:
        data (np.array): Normalized data.
    """

    data = (data - np.mean(data, axis=axis)) / np.std(data, axis=axis)
    return data


if __name__ == "__main__":
    pass
