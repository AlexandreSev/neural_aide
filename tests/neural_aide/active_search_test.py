#!/usr/bin/python
# coding: utf-8

import numpy as np
import tensorflow as tf
import unittest

from neural_aide import active_search
from alex_library import tf_utils

class ActiveSearchTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_find_k_most_uncertain(self):
        sess = tf.Session()
        nn = tf_utils.nn.NeuralNetwork(input_shape=1, hidden_sizes=[1],
                                       loss="binary_crossentropy",
                                       learning_rate=1.)
        x = np.vstack((
            np.ones((5, 1)),
            np.ones((5, 1)) * -1,
            ))
        y = np.array([1] * 5 + [0] * 5).reshape((-1, 1))
        nn.training(sess, x, y, n_epoch=50, display_step=10)
        print(sess.run(nn.params["weights"]["W0"]))
        print(sess.run(nn.params["biases"]["b0"]))
        


if __name__ == "__main__":
    unittest.main()
