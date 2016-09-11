# -*- coding: utf-8 -*-
import unittest
import dataset.array_dataset
import numpy as np

class ArrayDatasetTest(unittest.TestCase):

    def setUp(self):
        self.data = dataset.array_dataset.ArrayDataset()

    def test_empty(self):
        self.assertEqual(None, self.data.read())

    def test_set_data(self):
        self.data.set_data([[1,2],[4,5], [7,8]], np.array([1,2]).reshape((1,2)))
        result = self.data.read()
        self.assertTrue((np.array([1,4,7]).reshape((3,1)) == result[0]).all())
        self.assertTrue((np.array([1]) == result[1]).all())
        result = self.data.read()
        self.assertTrue((np.array([2,5,8]).reshape((3,1)) == result[0]).all())
        self.assertTrue((np.array([2]) == result[1]).all())
        result = self.data.read()
        self.assertEqual(None, result)

    def test_batch(self):
        self.data.set_data([[1,2],[4,5], [7,8]], np.array([1,2]).reshape((1,2)))
        result = self.data.read(2)
        self.assertTrue((np.array([[1,2],[4,5],[7,8]]) == result[0]).all())
        self.assertTrue((np.array([1,2]) == result[1]).all())
        result = self.data.read()
        self.assertEqual(None, result)

    def test_toomuch(self):
        self.data.set_data([[1,2],[4,5], [7,8]], np.array([1,2]).reshape((1,2)))
        result = self.data.read(3)
        self.assertEqual(None, result)
