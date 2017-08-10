import unittest
import level_set
import get_data
import math
from sympy import *
import numpy as np

image_matrix=get_data.get_image_data('/home/lza/Documents/level_set/gourd.bmp')

class TestLeveset(unittest.TestCase):
	"""docstring for test_leveset"""
	def test_random_init_region(self):
		random_region=level_set.random_init_region(image_matrix)
		self.assertTrue(1 in random_region)
	def test_narrowband(self):
		init_lsf=level_set.init_LSF(image_matrix,'random',10)
		narrowband=level_set.narrowband(init_lsf,np.ones(init_lsf.shape),1)
		self.assertTrue(1 in narrowband)
	def test_PDE_update(self):
		init_lsf=level_set.init_LSF(image_matrix,'random',10)
		result=level_set.PDE_update(init_lsf,2,5,0.04,1)
		self.assertTrue(1 in result)
		
if __name__ == '__main__':
	unittest.main()
		