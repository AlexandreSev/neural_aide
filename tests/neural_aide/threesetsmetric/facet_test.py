#!/usr/bin/python
# coding: utf-8

import numpy as np
import unittest

from neural_aide.threesetsmetric import facet

class FacetTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_2D_hyperplane_ref_is_not_visible(self):
        points = np.array([[1, 0], [0, 1]])
        ref = np.array([0, 0])
        test_facet = facet.Facet(points, ref)
        self.assertEqual(test_facet.is_visible(ref), False)

    def test_2D_hyperplane_a_point_is_visible(self):
        points = np.array([[1, 0], [0, 1]])
        ref = np.array([0, 0])
        test_point = np.array([1, 1])
        test_facet = facet.Facet(points, ref)
        self.assertEqual(test_facet.is_visible(test_point), True)

    def test_2D_hyperplane_coef(self):
        points = np.array([[1, 0], [0, 1]])
        ref = np.array([0, 0])
        test_facet = facet.Facet(points, ref)
        np.testing.assert_almost_equal(
            test_facet.coefs,
            np.array([[1. / np.sqrt(2), 1. / np.sqrt(2)]])
            )

    def test_3D_hyperplane_ref_is_not_visible(self):
        points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        ref = np.array([0, 0, 0])
        test_facet = facet.Facet(points, ref)
        self.assertEqual(test_facet.is_visible(ref), False)

    def test_3D_hyperplane_a_point_is_visible(self):
        points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        ref = np.array([0, 0, 0])
        test_point = np.array([1, 1, 1])
        test_facet = facet.Facet(points, ref)
        self.assertEqual(test_facet.is_visible(test_point), True)

    def test_3D_hyperplane_coef(self):
        points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        ref = np.array([0, 0, 0])
        test_facet = facet.Facet(points, ref)
        np.testing.assert_almost_equal(
            test_facet.coefs,
            np.array([[1. / np.sqrt(3), 1. / np.sqrt(3), 1. / np.sqrt(3)]])
            )

    def test_3D_facet_get_ridge_method(self):
        points = np.array([[1, 0, 0], [0, 1, 0.1], [0.1, 0.1, 1]])
        ref = np.array([0, 0, 0])
        test_facet = facet.Facet(points, ref)
        ridges = np.vstack(test_facet.get_ridges())
        answer = np.vstack([
            np.array([[0.1, 0.1, 1], [1, 0, 0]]),
            np.array([[0, 1, 0.1], [1, 0, 0]]),
            np.array([[0, 1, 0.1], [0.1, 0.1, 1]]),
            ])
        np.testing.assert_almost_equal(answer, ridges)

if __name__ == "__main__":
    unittest.main()
    