#!/usr/bin/python
# coding: utf-8

import numpy as np
import unittest

from neural_aide.threesetsmetric import posregion

class PosRegionTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_pos_region_compile_error(self):
        points = np.array([
            [0, 0],
            [0.1, 1],
            ])
        with self.assertRaises(ValueError):
            posregion.PosRegion(points)

    def test_2D_pos_region_good_number_of_facets(self):
        points = np.array([
            [0.2, 0.2],
            [0.1, 1],
            [1, 0.1],
            ])
        test = posregion.PosRegion(points)
        self.assertEqual(3, len(test.facets))

    def test_3D_pos_region_good_number_of_facets(self):
        points = np.array([
            [0.2, 0.2, 0.2],
            [0.1, 1, 0.1],
            [1, 0.1, 0.3],
            [0.3, 0.3, 1]
            ])
        test = posregion.PosRegion(points)
        self.assertEqual(4, len(test.facets))

    def test_2D_pos_region_contain(self):
        points = np.array([
            [0.2, 0.2],
            [0.1, 1],
            [1, 0.1],
            ])
        test = posregion.PosRegion(points)
        ref = np.array([0.5, 0.5])
        self.assertTrue(test.contain(ref))

    def test_2D_pos_region_not_contain(self):
        points = np.array([
            [0.2, 0.2],
            [0.1, 1],
            [1, 0.1],
            ])
        test = posregion.PosRegion(points)
        ref = np.array([1, 1])
        self.assertFalse(test.contain(ref))

    def test_2D_pos_region_add_vertex(self):
        points = np.array([
            [0.2, 0.2],
            [0.1, 1],
            [1, 0.1],
            ])
        test = posregion.PosRegion(points)
        test.add_vertex(np.array([1.1, 1.1]))
        ref = np.array([1, 1])
        self.assertTrue(test.contain(ref))

    def test_2D_pos_region_add_vertex(self):
        points = np.array([
            [0.2, 0.2],
            [0.1, 1],
            [1, 0.1],
            ])
        test = posregion.PosRegion(points)
        test.add_vertex(np.array([-0.1, -0.1]))
        ref = np.array([0, 0])
        self.assertTrue(test.contain(ref))

    def test_2D_pos_region_clean_vertex(self):
        points = np.array([
            [0.2, 0.2],
            [0.1, 1],
            [1, 0.1],
            ])
        test = posregion.PosRegion(points)
        test.add_vertex(np.array([-0.1, -0.1]))
        self.assertEqual(test.vertices.shape[0], 3)


if __name__ == "__main__":
    unittest.main()
