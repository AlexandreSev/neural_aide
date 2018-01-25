
import numpy as np
import logging
import unittest

from neural_aide.threesetsmetric import threesetsmanager


class ThreeSetsManagerTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_init(self):

        X_train = np.array([
            [0, 1],
            [1, 0],
            ])

        y_train = np.array([0, 1])

        X_val = np.array([
            [0, 0],
            [1, 1],
            ])

        tsm = threesetsmanager.ThreeSetsManager(X_train, y_train, X_val)

        self.assertEqual(tsm.dim, 2)

    def test_add_neg_sample_level_1(self):
        X_train = np.array([
            [0, 1],
            [1, 0],
            ])

        y_train = np.array([0, 1])

        X_val = np.array([
            [0, 0],
            [1, 1],
            ])

        tsm = threesetsmanager.ThreeSetsManager(X_train, y_train, X_val)

        to_add = 1
        tsm.add_sample(to_add, 0)

        self.assertEqual(tsm.neg_useful_samples, [])

    def test_add_pos_sample_create_neg_regions(self):
        X_train = np.array([
            [0, 1],
            [1, 0],
            ])

        y_train = np.array([0, 1])

        X_val = np.array([
            [0.01, 0.01],
            [1, 1],
            [0.5, 0.49],
            [0.25, 0.26],
            [0.75, 0.749999],
            ])

        tsm = threesetsmanager.ThreeSetsManager(X_train, y_train, X_val)

        tsm.add_sample(0, 1)

        self.assertTrue(tsm.neg_regions[0].contain(np.array([-0.5, 2])))

    def test_add_pos_sample_create_pos_regions_contain_good_pos_points(self):
        X_train = np.array([
            [0, 1],
            [1, 0],
            ])

        y_train = np.array([0, 1])

        X_val = np.array([
            [0.01, 0.01],
            [1, 1],
            [0.5, 0.49],
            [0.25, 0.26],
            [0.75, 0.749999],
            ])

        tsm = threesetsmanager.ThreeSetsManager(X_train, y_train, X_val)

        tsm.add_sample(0, 1)
        tsm.add_sample(3, 1)

        np.testing.assert_almost_equal(tsm.uncertain_samples,
                                       np.delete(X_val, [0, 4], 0))

    def test_add_pos_sample_create_pos_regions_contain_good_pos_points(self):
        X_train = np.array([
            [0, 1],
            [1, 0],
            ])

        y_train = np.array([0, 1])

        X_val = np.array([
            [0.01, 0.01],
            [1, 1],
            [0.5, 0.49],
            [0.25, 0.26],
            [0.75, 0.749999],
            ])

        tsm = threesetsmanager.ThreeSetsManager(X_train, y_train, X_val)

        tsm.add_sample(0, 1)
        tsm.add_sample(3, 1)

        np.testing.assert_almost_equal(tsm.uncertain_samples,
                                       np.delete(X_val, [0, 4], 0))

    def test_add_pos_sample_create_pos_regions_contain_a_point(self):
        X_train = np.array([
            [0, 1],
            [1, 0],
            ])

        y_train = np.array([0, 1])

        X_val = np.array([
            [0.01, 0.01],
            [1, 1],
            [0.5, 0.49],
            [0.25, 0.26],
            [0.75, 0.749999],
            ])

        tsm = threesetsmanager.ThreeSetsManager(X_train, y_train, X_val)

        tsm.add_sample(0, 1)
        tsm.add_sample(3, 1)

        self.assertTrue(tsm.pos_region.contain(X_val[2, :]))

    def test_add_pos_sample_create_pos_regions_not_contain_a_point(self):
        X_train = np.array([
            [0, 1],
            [1, 0],
            ])

        y_train = np.array([0, 1])

        X_val = np.array([
            [0.01, 0.01],
            [1, 1],
            [0.5, 0.49],
            [0.25, 0.26],
            [0.75, 0.749999],
            ])

        tsm = threesetsmanager.ThreeSetsManager(X_train, y_train, X_val)

        tsm.add_sample(0, 1)
        tsm.add_sample(3, 1)

        self.assertFalse(tsm.pos_region.contain(X_val[3, :]))

    def test_add_pos_sample_useless_for_pos_region_test_pos_region(self):
        X_train = np.array([
            [0, 1],
            [1, 0],
            ])

        y_train = np.array([0, 1])

        X_val = np.array([
            [0.01, 0.01],
            [1, 1],
            [0.5, 0.49],
            [0.25, 0.26],
            [0.75, 0.749999],
            ])

        tsm = threesetsmanager.ThreeSetsManager(X_train, y_train, X_val)

        tsm.add_sample(0, 1)
        tsm.add_sample(3, 1)
        tsm.add_sample(1, 1)

        self.assertEqual(tsm.pos_region.vertices.shape[0], 3)

    def test_add_pos_sample_useless_for_pos_region_test_pos_samples(self):
        X_train = np.array([
            [0, 1],
            [1, 0],
            ])

        y_train = np.array([0, 1])

        X_val = np.array([
            [0.01, 0.01],
            [1, 1],
            [0.5, 0.49],
            [0.25, 0.26],
            [0.75, 0.749999],
            [-0.5, 2],
            ])

        tsm = threesetsmanager.ThreeSetsManager(X_train, y_train, X_val)

        tsm.add_sample(0, 1)
        tsm.add_sample(3, 1)
        tsm.add_sample(1, 1)

        self.assertEqual(tsm.pos_samples.shape[0], 4)

    def test_add_pos_sample_useful_for_pos_region_test_pos_samples(self):
        X_train = np.array([
            [0, 1],
            [1, 0],
            ])

        y_train = np.array([0, 1])

        X_val = np.array([
            [0.01, 0.01],
            [1, 1],
            [0.5, 0.49],
            [0.25, 0.26],
            [0.75, 0.749999],
            [2, -0.001],
            ])

        tsm = threesetsmanager.ThreeSetsManager(X_train, y_train, X_val)

        tsm.add_sample(0, 1)
        tsm.add_sample(3, 1)
        tsm.add_sample(3, 1)

        self.assertTrue(tsm.pos_region.contain(np.array([1.5, 0.1])))

    def test_add_pos_sample_update_neg_regions(self):
        X_train = np.array([
            [0, 1],
            [1, 0],
            ])

        y_train = np.array([0, 1])

        X_val = np.array([
            [0.01, 0.01],
            [1, 1],
            [0.5, 0.49],
            [0.25, 0.26],
            [0.75, 0.749999],
            [2, -0.001],
            ])

        tsm = threesetsmanager.ThreeSetsManager(X_train, y_train, X_val)

        tsm.add_sample(0, 1)
        tsm.add_sample(3, 1)
        tsm.add_sample(3, 1)

        self.assertTrue(tsm.neg_regions[0].contain(np.array([-1.5, 2])))

    def test_add_neg_sample_useless_for_neg_regions(self):
        X_train = np.array([
            [0, 1],
            [1, 0],
            ])

        y_train = np.array([0, 1])

        X_val = np.array([
            [0.01, 0.01],
            [1, 1],
            [0.5, 0.49],
            [0.25, 0.26],
            [0.75, 0.749999],
            [2, -0.001],
            [-1.5, 2],
            ])

        tsm = threesetsmanager.ThreeSetsManager(X_train, y_train, X_val)

        tsm.add_sample(0, 1)
        tsm.add_sample(3, 1)
        tsm.add_sample(3, 1)
        tsm.add_sample(3, 1)

        self.assertEqual(len(tsm.neg_regions), 1)

    def test_add_neg_sample_replace_old_neg_regions(self):
        X_train = np.array([
            [0, 1],
            [1, 0],
            ])

        y_train = np.array([0, 1])

        X_val = np.array([
            [0.01, 0.01],
            [1, 1],
            [0.5, 0.49],
            [0.25, 0.26],
            [0.75, 0.749999],
            [2, -0.001],
            [-1.5, 2],
            [0.001, 0.999]
            ])

        tsm = threesetsmanager.ThreeSetsManager(X_train, y_train, X_val)

        tsm.add_sample(0, 1)
        tsm.add_sample(3, 1)
        tsm.add_sample(3, 1)
        tsm.add_sample(4, 0)

        self.assertEqual(len(tsm.neg_regions), 1)

    def test_add_neg_sample_create_new_neg_region(self):
        X_train = np.array([
            [0, 1],
            [1, 0],
            ])

        y_train = np.array([0, 1])

        X_val = np.array([
            [0.01, 0.01],
            [1, 1],
            [0.5, 0.49],
            [0.25, 0.26],
            [0.75, 0.749999],
            [2, -0.001],
            [-1.5, 2],
            [10.43, -34.2]
            ])

        tsm = threesetsmanager.ThreeSetsManager(X_train, y_train, X_val)

        tsm.add_sample(0, 1)
        tsm.add_sample(3, 1)
        tsm.add_sample(3, 1)
        tsm.add_sample(4, 0)

        self.assertEqual(len(tsm.neg_regions), 2)

    def test_update_regions_test_pos(self):
        X_train = np.array([
            [0, 1],
            [1, 0],
            ])

        y_train = np.array([0, 1])

        X_val = np.array([
            [0.01, 0.01],
            [1, 1],
            [0.5, 0.49],
            [0.25, 0.26],
            [0.75, 0.749999],
            [2, -0.001],
            [-1.5, 2],
            [10.43, -34.2],
            [1.001, -0.000001],
            [-0.5, 10],
            [0.5, -3],
            ])

        tsm = threesetsmanager.ThreeSetsManager(X_train, y_train, X_val)

        tsm.add_sample(0, 1)
        tsm.add_sample(3, 1)
        tsm.add_sample(3, 1)
        tsm.add_sample(4, 0)

        tsm.update_regions()

        self.assertEqual(tsm.pos_samples.shape[0], 6)

    def test_update_regions_test_neg(self):
        X_train = np.array([
            [0, 1],
            [1, 0],
            ])

        y_train = np.array([0, 1])

        X_val = np.array([
            [0.01, 0.01],
            [1, 1],
            [0.5, 0.49],
            [0.25, 0.26],
            [0.75, 0.749999],
            [2, -0.001],
            [-1.5, 2],
            [10.43, -34.2],
            [1.001, -0.000001],
            [-0.5, 10],
            [0.5, -3],
            ])

        tsm = threesetsmanager.ThreeSetsManager(X_train, y_train, X_val)

        tsm.add_sample(0, 1)
        tsm.add_sample(3, 1)
        tsm.add_sample(3, 1)
        tsm.add_sample(4, 0)

        tsm.update_regions()

        self.assertEqual(tsm.neg_samples.shape[0], 4)

    def test_update_regions_test_uncertain(self):
        X_train = np.array([
            [0, 1],
            [1, 0],
            ])

        y_train = np.array([0, 1])

        X_val = np.array([
            [0.01, 0.01],
            [1, 1],
            [0.5, 0.49],
            [0.25, 0.26],
            [0.75, 0.749999],
            [2, -0.001],
            [-1.5, 2],
            [10.43, -34.2],
            [1.001, -0.000001],
            [-0.5, 10],
            [0.5, -3],
            ])

        tsm = threesetsmanager.ThreeSetsManager(X_train, y_train, X_val)

        tsm.add_sample(0, 1)
        tsm.add_sample(3, 1)
        tsm.add_sample(3, 1)
        tsm.add_sample(4, 0)

        tsm.update_regions()

        self.assertEqual(tsm.uncertain_samples.shape[0], 3)

    def test_get_new_sets(self):
        X_train = np.array([
            [0, 1],
            [1, 0],
            ])

        y_train = np.array([0, 1])

        X_val = np.array([
            [0.01, 0.01],
            [1, 1],
            [0.5, 0.49],
            [0.25, 0.26],
            [0.75, 0.749999],
            [2, -0.001],
            [-1.5, 2],
            [10.43, -34.2],
            [1.001, -0.000001],
            [-0.5, 10],
            [0.5, -3],
            ])

        tsm = threesetsmanager.ThreeSetsManager(X_train, y_train, X_val)

        tsm.add_sample(0, 1)
        tsm.add_sample(3, 1)
        tsm.add_sample(3, 1)
        tsm.add_sample(4, 0)

        tsm.update_regions()

        X_train, y_train, X_val = tsm.get_new_sets()

        logging.debug(" ########### Here is X_train: %s" % (X_train,))
        logging.debug(" ########### Here is y_train: %s" % (y_train,))

        answer = np.array([1, 3, 10])
        np.testing.assert_almost_equal(X_val, answer)


if __name__ == "__main__":
    logging.basicConfig(level=10,
                        format='%(asctime)s %(levelname)s: %(message)s')
    unittest.main()
