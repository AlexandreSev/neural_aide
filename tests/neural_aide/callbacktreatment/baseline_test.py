#!/usr/bin/python
# coding: utf-8

import numpy as np
import unittest
import logging

from neural_aide.callbacktreatment import baseline


class samplingTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_f1score_always_0(self):
        y = np.array([0, 0, 0, 0, 1]).reshape((-1, 1))
        pred = np.zeros(y.shape)

        self.assertEqual(baseline.f1_score(y, pred), 0)

    def test_f1score_1(self):
        y = np.array([0, 0, 0, 0, 1]).reshape((-1, 1))
        pred = np.zeros(y.shape)
        pred[-1] = 1

        self.assertEqual(baseline.f1_score(y, pred), 1)

    def test_f1score_simple(self):
        y = np.array([0, 0, 0, 0, 1]).reshape((-1, 1))
        pred = np.zeros(y.shape)
        pred[-1] = 1
        pred[0] = 1

        self.assertEqual(baseline.f1_score(y, pred), 2./3.)

    def test_create_naive_score(self):
        callback = {"samples": range(10)}

        labels = np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 1]).reshape((-1, 1))

        answer = baseline.create_naive_score(callback, labels)
        np.testing.assert_almost_equal(answer, [7./8, 14./15, 14./15., 1, 1, 1,
                                                1, 1, 1])

   
if __name__ == "__main__":
    logging.basicConfig(level=10,
                        format='%(asctime)s %(levelname)s: %(message)s')
    unittest.main()
    