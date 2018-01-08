#!/usr/bin/python
# coding: utf-8

import logging
import numpy as np
import time

def kmin(l1, l2, k):
    """
    Return the k smaller elements of two lists of tuples, sorted by their first
        element. If there are not enough elements, return all of them, sorted.
    Params:
        l1 (list of tuples): first list. Must be sorted.
        l2 (list of tuples): second list. Must be sorted.
        k (integer): number of elements to return.

    Return:
        (list): k smaller elements
    """
    results = []
    while len(results) < k:
        if l1 == []:
            if l2 == []:
                return results
            else:
                results.append(l2[0])
                l2 = l2[1:]
        else:
            if l2 == []:
                results.append(l1[0])
                l1 = l1[1:]
            else:
                if l1[0][0] > l2[0][0]:
                    results.append(l2[0])
                    l2 = l2[1:]
                else:
                    results.append(l1[0])
                    l1 = l1[1:]
    return results


def find_k_most_uncertain(nn, sess, X, batch_size=None, k=2,
                          pool_size=None):
    """
    Find the most uncertain sample for a neural network in a database.
    Params:
        nn (ActiveNeuralNetwork): The neural network evaluated.
        sess (tf.Session): The session associated to the nn.
        X (np.array): The database used.
        batch_size (integer): size of the batch during evaluation.
        k (integer): number of samples to be returned.
        pool_size (integer): Size of the pool considered to find the most
            uncertain point. If None, the whole X is used.

    Return:
        (list of integers): k most uncertain indices
    """

    # Initialization
    results = []
    i = -1

    if batch_size is None:
        _batch_size = X.shape[0]
    else:
        _batch_size = batch_size

    # Create the pool
    order = np.arange(X.shape[0])

    if pool_size is not None:
        order = np.random.choice(X.shape[0], 
                                 min(X.shape[0], pool_size),
                                 replace=False)
        X_pool = X[order, :].copy()
    else:
        X_pool = X.copy()

    # Loop over the batches
    for i in range(X_pool.shape[0]/_batch_size):

        feed_dict = {}
        feed_dict[nn.input_tensor] = X_pool[i * _batch_size:
                                            (i+1) * _batch_size]

        # Predict the batch
        pred = sess.run(nn.prediction, feed_dict=feed_dict)

        # The most uncertain is the closest to 0.5
        pred = np.abs(0.5 - pred).reshape(-1)

        # Sort it by uncertainty
        batch_order = np.argsort(pred)
        pred = pred[batch_order]

        # Create associated indices
        to_zip = order[range(i * _batch_size, i * _batch_size + pred.shape[0])]
        to_zip = to_zip[batch_order]

        results = kmin(results, zip(pred, to_zip), k)

    # Last uncomplete batch
    feed_dict = {}
    feed_dict[nn.input_tensor] = X_pool[(i+1) * _batch_size:]

    # Predict the last batch
    pred = sess.run(nn.prediction, feed_dict=feed_dict)

    # Sort it by uncertainty
    pred = np.abs(0.5 - pred).reshape(-1)
    batch_order = np.argsort(pred)
    pred = pred[batch_order]

    # Create associated indices
    to_zip = order[(i + 1) * _batch_size:]
    to_zip = to_zip[batch_order]

    results = kmin(results, zip(pred, to_zip), k)

    return [i[1] for i in results]


def initial_sampling(y):
    """
    Pick randomly two samples which have a different label
    Params:
        y (np.array): labels

    Return:
        (list of integers): indices of picked samples
    """
    samples = list(np.random.randint(0, len(y), 2))
    while len(np.unique(y[samples] > 0.5)) != 2:
        samples = list(np.random.randint(0, len(y), 2))
    return samples


def uncertainty_sampling(nn_main, sess_main, X, pool_size=None):
    """
    Find the next sample with uncertainty sampling.
    Params:
        nn_main (ActiveNeuralNetwork): Main nn of the active search.
        sess_main (tf.Session): sessions associated with the nn.
        X (np.array): Database used during the active search.
        pool_size (integer): Size of the pool considered to find the most
            uncertain point. If None, the whole X is used.

    Return:
        (integer) indice of the new sample
    """

    temp = find_k_most_uncertain(nn_main, sess_main, X, k=1,
                                 pool_size=pool_size)

    return temp[0]
