# -*- coding: utf-8 -*-
import unittest
import minicaffe.dataset.array_dataset as array_dataset
import numpy as np

class ArrayDatasetTest(unittest.TestCase):

    def setUp(self):
        self.data = array_dataset.ArrayDataset()

    def test_empty(self):
        self.assertEqual(None, self.data.read())

    def test_set_data(self):
        self.data.set_data([[1, 4, 7],[2, 5, 8]], np.array([1,2]).reshape((2)))
        result = self.data.read()
        self.assertTrue((np.array([1,4,7]).reshape((1,3)) == result[0].mutable_data()).all())
        self.assertTrue((np.array([1]) == result[1].mutable_data()).all())
        result = self.data.read()
        self.assertTrue((np.array([2,5,8]).reshape((1,3)) == result[0].mutable_data()).all())
        self.assertTrue((np.array([2]) == result[1].mutable_data()).all())
        result = self.data.read()
        self.assertEqual(None, result)

    def test_batch(self):
        self.data.set_data([[1,2],[4,5],[7,8]], np.array([1,2,3]).reshape((3)))
        result = self.data.read(2)
        self.assertTrue((np.array([[1,2],[4,5]]) == result[0].mutable_data()).all())
        self.assertTrue((np.array([1,2]) == result[1].mutable_data()).all())
        result = self.data.read(5)
        self.assertEqual(None, result)

    def test_toomuch(self):
        self.data.set_data([[1,4,7],[2,5,8]], np.array([1,2]).reshape((1,2)))
        result = self.data.read(3)
        self.assertEqual(None, result)
