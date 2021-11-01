import unittest
import sparseSpACE
import sparseSpACE.HP_Optimization as hpo
import numpy as np
import math

class TestHP_Optimization(unittest.TestCase):
	def test_HPO(self):
		OC = hpo.Optimize_Classification(data_name = "moons")
		HPO = hpo.HP_Optimization(OC.pea_classification, OC.classification_space)
		#test beta
		beta_1 = HPO.get_beta(1)
		self.assertAlmostEqual(beta_1, 34.74985580606742)
		#test cov
		v1 = np.array([1, 2])
		v2 = np.array([3, 4])
		res = HPO.cov(v1, v2)
		self.assertAlmostEqual(res, math.exp(-16))
		#test cov_matrix
		C_x=np.array([[1, 2], [3, 4]])
		K = HPO.get_cov_matrix(C_x, 2)
		control_K = np.array([[1, math.exp(-16)], [math.exp(-16), 1]])
		np.testing.assert_almost_equal(K, control_K)
		#test cov_vector
		C_x=np.array([[1, 2], [5, 6]])
		x = np.array([3, 4])
		v = HPO.get_cov_vector(C_x, x, 2)
		control_v = np.array([math.exp(-16), math.exp(-64)])
		np.testing.assert_almost_equal(v, control_v)
		simple_function = lambda x: x[0]+x[1]
		simple_space = [["list", 0.42, 1, 2.5, 3.3], ["list", 2, 4]]
		HPO_simple = hpo.HP_Optimization(simple_function, simple_space)
		#test GO
		res_GO = HPO_simple.perform_GO()
		np.testing.assert_almost_equal(res_GO[0], [3.3, 4])
		self.assertAlmostEqual(res_GO[1], 7.3)
		##test RO - should I even test RO?? It's usually not that close to the optimum
		message = "The optimum found by perform_RO is not close enough to the actual optimum"
		res_RO = HPO_simple.perform_RO(15)
		np.testing.assert_almost_equal(res_RO[0], [3.3, 4], 1, message)
		self.assertAlmostEqual(res_RO[1], 7.3, 2)
	def simple_function(self, x):
		return x[0]+x[1]


unittest.main()
