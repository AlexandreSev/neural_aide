#!/usr/bin/python
# coding: utf-8

import numpy as np
import unittest
import logging

from neural_aide.threesetsmetric.evaluating_points import EvaluatingGridPoints

class EvaluatingGridPointsTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_2D_simple(self):
        lb = [0, 0]
        ub = [1, 1]
        n_points = 4
        test = EvaluatingGridPoints(n_points, lb, ub)
        answer = np.array([
            [0, 0], 
            [1, 0],
            [0, 1],
            [1, 1],
            ])
        np.testing.assert_almost_equal(test, answer)

    def test_2D_negative(self):
        lb = [-1, 0]
        ub = [1, 1]
        n_points = 4
        test = EvaluatingGridPoints(n_points, lb, ub)
        answer = np.array([
            [-1, 0], 
            [1, 0],
            [-1, 1],
            [1, 1],
            ])
        np.testing.assert_almost_equal(test, answer)

    def test_2D_n_points_approx(self):
        lb = [-1, 0]
        ub = [1, 1]
        n_points = 6
        test = EvaluatingGridPoints(n_points, lb, ub)
        answer = np.array([
            [-1, 0], 
            [1, 0],
            [-1, 1],
            [1, 1],
            ])
        np.testing.assert_almost_equal(test, answer)

    def test_2D_more_points(self):
        lb = [-5, 0]
        ub = [0, 10]
        n_points = 40
        test = EvaluatingGridPoints(n_points, lb, ub)
        answer = np.array([
            [-5, 0], 
            [-4, 0],
            [-3, 0],
            [-2, 0],
            [-1, 0],
            [0, 0], 
            [-5, 2], 
            [-4, 2],
            [-3, 2],
            [-2, 2],
            [-1, 2],
            [0, 2], 
            [-5, 4], 
            [-4, 4],
            [-3, 4],
            [-2, 4],
            [-1, 4],
            [0, 4], 
            [-5, 6], 
            [-4, 6],
            [-3, 6],
            [-2, 6],
            [-1, 6],
            [0, 6], 
            [-5, 8], 
            [-4, 8],
            [-3, 8],
            [-2, 8],
            [-1, 8],
            [0, 8], 
            [-5, 10], 
            [-4, 10],
            [-3, 10],
            [-2, 10],
            [-1, 10],
            [0, 10], 
            ])
        np.testing.assert_almost_equal(test, answer)

    def test_3D(self):
        lb = [-1, 0, 0]
        ub = [1, 1, 1]
        n_points = 8
        test = EvaluatingGridPoints(n_points, lb, ub)
        answer = np.array([
            [-1, 0, 0], 
            [1, 0, 0],
            [-1, 1, 0],
            [1, 1, 0],
            [-1, 0, 1], 
            [1, 0, 1],
            [-1, 1, 1],
            [1, 1, 1],
            ])
        np.testing.assert_almost_equal(test, answer)

if __name__ == "__main__":
    logging.basicConfig(level=10,
                        format='%(asctime)s %(levelname)s: %(message)s')
    unittest.main()
