#!/usr/bin/python
# coding: utf-8

import numpy as np
import tensorflow as tf
import unittest
import logging

from neural_aide import sampling
from neural_aide.active_nn import ActiveNeuralNetwork
from alex_library.tf_utils import utils

class samplingTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_kmin_simple(self):
        l1 = [(1, 0), (2, 1), (3, 2), (4, 3)]
        l2 = [(1, 4), (4, 5), (5, 6)]
        answer = sampling.kmin(l1, l2, 3)
        self.assertEqual(answer, [(1, 0), (1, 4), (2, 1)])

    def test_kmin_not_enough_elements(self):
        l1 = [(1, 0), (2, 1), (3, 2), (4, 3)]
        l2 = [(1, 4), (4, 5), (5, 6)]
        answer = sampling.kmin(l1, l2, 113)
        self.assertEqual(answer, [(1, 0), (1, 4), (2, 1), (3, 2), (4, 3),
                                  (4, 5), (5, 6)])

    def test_k_most_uncertain_simple(self):
        nn = ActiveNeuralNetwork(
            input_shape=2, hidden_shapes=[32, 32, 1],
            loss="binary_crossentropy"
            )
        sess = tf.Session()
        weights_path = ("/Users/alex/Documents/LIX-PHD/experiments/clean_activ"
                        + "e_nn/ressources/tests/sampling/weights_main.pckl")

        utils.loader(nn.params, sess, weights_path)

        X = np.array([
            [0.25, 1.6],
            [0.2, 1.65],
            [0.27, 1.55],
            [0.25, 1.5]])

        pred = sess.run(nn.prediction, feed_dict={nn.input_tensor: X})

        logging.debug("Here is pred: %s" % (pred,))

        answer = sampling.find_k_most_uncertain(nn, sess, X, k=1)

        self.assertEqual(answer, [3])

    def test_k_most_uncertain_small_batch(self):
        nn = ActiveNeuralNetwork(
            input_shape=2, hidden_shapes=[32, 32, 1],
            loss="binary_crossentropy"
            )
        sess = tf.Session()
        weights_path = ("/Users/alex/Documents/LIX-PHD/experiments/clean_activ"
                        + "e_nn/ressources/tests/sampling/weights_main.pckl")

        utils.loader(nn.params, sess, weights_path)

        X = np.array([
            [0.25, 1.6],
            [0.2, 1.65],
            [0.27, 1.55],
            [0.25, 1.5]])

        pred = sess.run(nn.prediction, feed_dict={nn.input_tensor: X})

        logging.debug("Here is pred: %s" % (pred,))

        answer = sampling.find_k_most_uncertain(nn, sess, X, k=1, batch_size=2)

        self.assertEqual(answer, [3])

    def test_k_most_uncertain_pool(self):
        nn = ActiveNeuralNetwork(
            input_shape=2, hidden_shapes=[32, 32, 1],
            loss="binary_crossentropy"
            )
        sess = tf.Session()
        weights_path = ("/Users/alex/Documents/LIX-PHD/experiments/clean_activ"
                        + "e_nn/ressources/tests/sampling/weights_main.pckl")

        utils.loader(nn.params, sess, weights_path)

        X = np.array([
            [0.25, 1.6],
            [0.2, 1.65],
            [0.27, 1.55],
            [0.25, 1.5]])

        pred = sess.run(nn.prediction, feed_dict={nn.input_tensor: X})

        logging.debug("Here is pred: %s" % (pred,))

        np.random.seed(42)

        answer = sampling.find_k_most_uncertain(nn, sess, X, k=1, pool_size=2)

        self.assertEqual(answer, [3])

    def test_k_most_uncertain_pool_and_batch(self):
        nn = ActiveNeuralNetwork(
            input_shape=2, hidden_shapes=[32, 32, 1],
            loss="binary_crossentropy"
            )
        sess = tf.Session()
        weights_path = ("/Users/alex/Documents/LIX-PHD/experiments/clean_activ"
                        + "e_nn/ressources/tests/sampling/weights_main.pckl")

        utils.loader(nn.params, sess, weights_path)

        X = np.array([
            [0.25, 1.6],
            [0.2, 1.65],
            [0.27, 1.55],
            [0.25, 1.5]])

        pred = sess.run(nn.prediction, feed_dict={nn.input_tensor: X})

        logging.debug("Here is pred: %s" % (pred,))

        np.random.seed(42)

        answer = sampling.find_k_most_uncertain(nn, sess, X, k=1, pool_size=3,
                                                batch_size=2)

        self.assertEqual(answer, [3])

    def test_initial_sampling(self):
        y = np.array([0, 0, 0, 0, 1, 0 ])
        np.random.seed(42)
        answer = sampling.initial_sampling(y)
        self.assertEqual(answer, [3, 4])

    def test_uncertatinty_sampling_pool(self):
        nn = ActiveNeuralNetwork(
            input_shape=2, hidden_shapes=[32, 32, 1],
            loss="binary_crossentropy"
            )
        sess = tf.Session()
        weights_path = ("/Users/alex/Documents/LIX-PHD/experiments/clean_activ"
                        + "e_nn/ressources/tests/sampling/weights_main.pckl")

        utils.loader(nn.params, sess, weights_path)

        X = np.array([
            [0.25, 1.6],
            [0.2, 1.65],
            [0.27, 1.55],
            [0.25, 1.5]])

        pred = sess.run(nn.prediction, feed_dict={nn.input_tensor: X})

        logging.debug("Here is pred: %s" % (pred,))

        np.random.seed(42)

        answer = sampling.uncertainty_sampling(nn, sess, X, pool_size=3)

        self.assertEqual(answer, 3)


if __name__ == "__main__":
    logging.basicConfig(level=10,
                        format='%(asctime)s %(levelname)s: %(message)s')
    unittest.main()
    