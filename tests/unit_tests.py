import unittest
import numpy as np
import math

from utils import *
from environment import DTMC

class TestEntropyProbabilities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set parameters
        cls.q = 5e-3
        cls.eta = 1
        cls.zeta = 1e-4
        cls.epsilon = 0.1
        cls.rho = 5e-2
        cls.alpha = 0.02
        cls.R = 10
        cls.mm = DTMC(cls.q, cls.eta)
        cls.pi = cls.mm.pi
        cls.A = cls.mm.A
        cls.K_values = [5, 10, 20, 40, 60]
        cls.x_symbols = np.array([0, 1])
        cls.y_symbols = np.array([0, 1, 2, 3])
        cls.sim_delta_values = [0, 1, 5, 10]  # sample delta values to test
    
    def test_ps_matches_binomial(self):
        """ps computed with PoiBin matches direct binomial formula for identical probs"""
        for K in self.K_values:
            m = math.floor(self.rho * np.pi * self.R * K)
            if m == 0:
                continue
            ps_poibin = ps(m, self.zeta, self.epsilon)
            p = self.zeta * (1 - self.epsilon)
            ps_binomial = m * p * (1 - p)**(m - 1)
            self.assertAlmostEqual(ps_poibin, ps_binomial, places=10,
                msg=f"ps mismatch for m={m}, K={K}")
    
    def test_prob_y_range(self):
        """prob_y returns valid probability values in [0,1] for all y and K"""
        for K in self.K_values:
            for y in self.y_symbols:
                val = p_y(y, self.pi, K, self.R, self.alpha)
                self.assertTrue(0 <= val <= 1,
                                f"prob_y out of bounds for y={y}, K={K}: {val}")
    
    def test_prob_y_given_x_0_and_prob_x_given_y_0(self):
        """prob_y_given_x_0 and prob_x_given_y_0 return valid probabilities and consistent"""
        for K in self.K_values:
            for y in self.y_symbols:
                sum_pyx = 0
                for x in self.x_symbols:
                    pyx = prob_y_given_x_0(x, y, K, self.R, self.alpha)
                    self.assertTrue(0 <= pyx <= 1,
                                    f"prob_y_given_x_0 out of bounds x={x}, y={y}, K={K}: {pyx}")
                    sum_pyx += pyx
                # sum of prob_y_given_x_0 over x should be close to 1 for each y (normalized)
                self.assertAlmostEqual(sum_pyx, 1, places=10,
                                       msg=f"Sum prob_y_given_x_0 != 1 for y={y}, K={K}")

                # Check prob_x_given_y_0 normalization
                sum_pxy = 0
                for x in self.x_symbols:
                    pxy = prob_x_given_y_0(x, y, self.pi, K, self.R, self.alpha)
                    self.assertTrue(0 <= pxy <= 1,
                                    f"prob_x_given_y_0 out of bounds x={x}, y={y}, K={K}: {pxy}")
                    sum_pxy += pxy
                self.assertAlmostEqual(sum_pxy, 1, places=10,
                                       msg=f"Sum prob_x_given_y_0 != 1 for y={y}, K={K}")
    
    def test_prob_x_given_y_delta_normalization(self):
        """prob_x_given_y_delta returns valid distribution summing to 1"""
        for K in self.K_values:
            for y in self.y_symbols:
                p_x_given_y_0_arr = np.array([prob_x_given_y_0(x, y, self.pi, K, self.R, self.alpha) for x in self.x_symbols])
                for delta in self.sim_delta_values:
                    p_x_given_y_delta = prob_x_given_y_delta(p_x_given_y_0_arr, self.A, delta)
                    # Should be probabilities summing to 1
                    self.assertTrue(np.all(p_x_given_y_delta >= 0),
                                    f"Negative prob_x_given_y_delta for delta={delta}")
                    self.assertAlmostEqual(np.sum(p_x_given_y_delta), 1, places=10,
                                           msg=f"prob_x_given_y_delta sum != 1 for delta={delta}")
    
    def test_h_y_delta_values(self):
        """h_y_delta returns non-negative entropy values"""
        for K in self.K_values:
            for y in self.y_symbols:
                p_x_given_y_0_arr = np.array([prob_x_given_y_0(x, y, self.pi, K, self.R, self.alpha) for x in self.x_symbols])
                for delta in self.sim_delta_values:
                    h_val = h_y_delta(p_x_given_y_0_arr, self.A, delta, self.x_symbols)
                    self.assertTrue(h_val >= 0,
                                    f"Negative entropy h_y_delta for delta={delta}")
                    # Entropy should not exceed 1 bit for binary symbols
                    self.assertTrue(h_val <= 1 + 1e-8,
                                    f"Entropy h_y_delta too high for delta={delta}: {h_val}")
    

if __name__ == "__main__":
    unittest.main()
