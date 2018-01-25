#!/usr/bin/python
# coding: utf-8

import unittest
import logging

from neural_aide import qdbsampling


class qdb_samplingTest(unittest.TestCase):

    def setUp(self):
        pass

if __name__ == "__main__":
    logging.basicConfig(level=10,
                        format='%(asctime)s %(levelname)s: %(message)s')
    unittest.main()
    