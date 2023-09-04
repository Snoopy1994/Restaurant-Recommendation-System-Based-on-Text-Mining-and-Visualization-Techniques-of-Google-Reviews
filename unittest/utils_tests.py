import unittest
import numpy as np

import sys
sys.path.append('..')
import utils

class TestLabelChange(unittest.TestCase):

	def test_bool_input(self):
		A = np.array([False, True, True, True, False, False, True])
		X = utils.label_change(A)
		self.assertTrue(np.all(X == np.r_[1, 4, 6]))

	def test_number_input(self):
		A = np.array([0, 1, 1, 1, 3, 3, 2])
		X = utils.label_change(A)
		self.assertTrue(np.all(X == np.r_[1, 4, 6]))

	def test_word_input(self):
		A = np.array(['A', 'B', 'B', 'B', 'C', 'C', 'D'])
		X = utils.label_change(A)
		self.assertTrue(np.all(X == np.r_[1, 4, 6]))

class TestSlidingOperation(unittest.TestCase):

	def test_sliding_sum(self):
		A = np.array([0, 3, 3, 3, 3, 3, 0])
		X = utils.sliding_sum(A, 1)
		self.assertTrue(np.all(X == np.r_[3, 6, 9, 9, 9, 6, 3]))

	def test_sliding_mean(self):
		A = np.array([0, 3, 3, 3, 3, 3, 0])
		X = utils.sliding_mean(A, 1)
		self.assertTrue(np.all(X == np.r_[1, 2, 3, 3, 3, 2, 1]))

if __name__ == '__main__':
	unittest.main()