#!/usr/bin/python
# coding: utf-8

import numpy as np
import logging
import unittest

from neural_aide.threesetsmetric import negregion, posregion

class NegRegionTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_neg_region_compile_error(self):
        neg_point = np.array([
            [0, 0],
            ])
        pos_samples = np.array([
            [1, 1],
            ])
        with self.assertRaises(ValueError):
            negregion.NegRegion(neg_point, pos_samples)

    def test_2D_neg_region_from_points_contain(self):
        neg_point = np.array([-0.1, -0.1])
        pos_points = np.array([
            [1, 0.1],
            [0.1, 1],
            ])
        test = negregion.NegRegion(neg_point, pos_points)
        ref = np.array([-0.5, -0.5])
        self.assertTrue(test.contain(ref))

    def test_2D_neg_region_from_region_contain(self):
        neg_point = np.array([-0.1, -0.1])
        pos_points = np.array([
            [1, 0.1],
            [0.1, 1],
            [1.1, 1.1],
            ])
        pos_region = posregion.PosRegion(pos_points)
        test = negregion.NegRegion(neg_point, None, pos_region)
        ref = np.array([-0.5, -0.5])
        self.assertTrue(test.contain(ref))

    def test_2D_neg_region_from_region_contain_harder(self):
        neg_point = np.array([-0.1, -0.1])
        pos_points = np.array([
            [1, 0.1],
            [0.1, 1],
            [0.5, 0.5],
            ])
        pos_region = posregion.PosRegion(pos_points)
        test = negregion.NegRegion(neg_point, None, pos_region)
        ref = np.array([-0.5, -0.5])
        self.assertTrue(test.contain(ref))

    def test_2D_neg_region_from_region_not_contain(self):
        neg_point = np.array([-0.1, -0.1])
        pos_points = np.array([
            [1, 0.1],
            [0.1, 1],
            [1.1, 1.1],
            ])
        pos_region = posregion.PosRegion(pos_points)
        test = negregion.NegRegion(neg_point, None, pos_region)
        ref = np.array([0, 0])
        self.assertFalse(test.contain(ref))

    def test_2D_neg_region_add_vertex(self):
        neg_point = np.array([-0.1, -0.1])
        pos_points = np.array([
            [1, 0.1],
            [0.1, 1],
            [1.1, 1.1],
            ])
        pos_region = posregion.PosRegion(pos_points)
        test = negregion.NegRegion(neg_point, None, pos_region)
        test.add_vertex(np.array([-1, 1]))
        ref = np.array([0.5, -1])
        self.assertTrue(test.contain(ref))


if __name__ == "__main__":
    logging.basicConfig(level=10, 
                        format='%(asctime)s %(levelname)s: %(message)s')
    unittest.main()
