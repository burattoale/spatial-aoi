"""
Python translation of stif_H_agnostic_new.m
Computes spatial entropy H with H-agnostic approach for spatial-AoI systems
"""

import numpy as np
from typing import Tuple, List
from numba import jit


def stif_h_agnostic_new(R: float, K: int, alpha: float, rho: float, zeta: float, 
                        q: float, beta: float, epsilon: float = 0.0) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Python translation of MATLAB function stif_H_agnostic_new
    
    Parameters:
    -----------
    R : float
        Unit radius
    K : int
        Number of spatial zones/buckets
    alpha : float
        Path loss exponent for lambda calculation
    rho : float
        Node density
    zeta : float
        Transmission probability
    q : float
        Markov chain parameter (transition probability)
    beta : float
        Markov chain parameter
    epsilon : float, optional
        Transmission erasure probability (default 0.0)
        
    Returns:
    --------
    H : float
        Average uncertainty (entropy)
    cdfH : np.ndarray
        CDF of entropy values
    gammaVec : np.ndarray
        Gamma vector for CDF calculation
    """
    zeta = zeta * (1 - epsilon)  # Adjust zeta for packet loss
    # Overall range
    R_max = R * K
    
    # Probability that a received update is from i-th zone, i = 1,...,K
    k_vec = np.arange(1, K + 1)
    
    # Probability that an update from i-th zone holds true value
    lambda_vec = 1.0 / ((1 + (k_vec - 1) * R) ** alpha)
    
    # Probability for node to be in one of the areas
    p_area = (2 * (k_vec - 1) + 1) / (K ** 2)
    
    # Probability of successfully receiving an update
    n = int(np.ceil(np.pi * rho * R_max ** 2))
    ps = zeta * (1 - epsilon) * n * (1 - zeta * (1 - epsilon)) ** (n - 1)
    
    # Markov process
    pi0 = beta / (1 + beta)
    pi1 = 1 - pi0
    
    # Transition matrix
    A = np.array([[1 - q, q],
                  [beta * q, 1 - beta * q]])
    
    # Find maxAge s.t. P(age <= maxAge) = 0.99
    max_age = 5000
    age_vec = np.arange(0, max_age + 1)
    
    # Initialize probability matrices
    # Series of 2x2 matrices for p(Xn | Yn, age, distance)
    p_xn_given_yn_age_dist = np.zeros((2, 2, len(age_vec), K))
    
    # Step 1: Compute p(yn, dn | xn) for each distance
    p_yn_given_xn_dn = np.zeros((2, 2, K))  # [yn, xn, dn]
    p_yn_dn_given_xn = np.zeros((2, 2, K))
    
    for kk in range(K):
        # p(yn | xn, dn) matrix
        # [p(0|0) p(1|0)]
        # [p(0|1) p(1|1)]
        p_yn_given_xn_dn[:, :, kk] = np.array([
            [lambda_vec[kk], 1 - lambda_vec[kk]],
            [1 - lambda_vec[kk], lambda_vec[kk]]
        ])
        p_yn_dn_given_xn[:, :, kk] = ps * p_area[kk] * p_yn_given_xn_dn[:, :, kk]
    
    # Step 1b: Bayes rule to get p(xn | yn, dn)
    p_xn_given_yn_dn = np.zeros((2, 2, K))
    
    for kk in range(K):
        # p(X=0 | Y=0, D=kk)
        denominator_y0 = pi0 * p_yn_dn_given_xn[0, 0, kk] + pi1 * p_yn_dn_given_xn[0, 1, kk]
        if denominator_y0 > 0:
            p_xn_given_yn_dn[0, 0, kk] = pi0 * p_yn_dn_given_xn[0, 0, kk] / denominator_y0
            p_xn_given_yn_dn[0, 1, kk] = pi1 * p_yn_dn_given_xn[0, 1, kk] / denominator_y0
        
        # p(X=0 | Y=1, D=kk)  
        denominator_y1 = pi0 * p_yn_dn_given_xn[1, 0, kk] + pi1 * p_yn_dn_given_xn[1, 1, kk]
        if denominator_y1 > 0:
            p_xn_given_yn_dn[1, 0, kk] = pi0 * p_yn_dn_given_xn[1, 0, kk] / denominator_y1
            p_xn_given_yn_dn[1, 1, kk] = pi1 * p_yn_dn_given_xn[1, 1, kk] / denominator_y1
    
    # Step 2: Compute p(dn | yn)
    p_yn_given_dn = np.zeros((2, K))
    
    for kk in range(K):
        p_yn_given_dn[0, kk] = p_yn_given_xn_dn[0, 0, kk] * pi0 + p_yn_given_xn_dn[0, 1, kk] * pi1
        p_yn_given_dn[1, kk] = p_yn_given_xn_dn[1, 0, kk] * pi0 + p_yn_given_xn_dn[1, 1, kk] * pi1
    
    # Step 2b: Bayes to compute p(dn | yn)
    p_dn_given_yn = np.zeros((K, 2))
    
    for kk in range(K):
        p_dn_given_yn[kk, 0] = p_yn_given_dn[0, kk] * p_area[kk] * ps
        p_dn_given_yn[kk, 1] = p_yn_given_dn[1, kk] * p_area[kk] * ps
    
    # Normalize
    sum_y0 = np.sum(p_dn_given_yn[:, 0])
    sum_y1 = np.sum(p_dn_given_yn[:, 1])
    
    if sum_y0 > 0:
        p_dn_given_yn[:, 0] /= sum_y0
    if sum_y1 > 0:
        p_dn_given_yn[:, 1] /= sum_y1
    
    # Step 3: Average over Dn, computing p(xn | yn)
    p_xn_given_yn = np.zeros((2, 2))
    
    p_xn_given_yn[0, 0] = np.sum(p_xn_given_yn_dn[0, 0, :] * p_dn_given_yn[:, 0])
    p_xn_given_yn[0, 1] = np.sum(p_xn_given_yn_dn[0, 1, :] * p_dn_given_yn[:, 0])
    p_xn_given_yn[1, 0] = np.sum(p_xn_given_yn_dn[1, 0, :] * p_dn_given_yn[:, 1])
    p_xn_given_yn[1, 1] = np.sum(p_xn_given_yn_dn[1, 1, :] * p_dn_given_yn[:, 1])
    
    # Find p(xn | yn, age) via n-step evolution
    p_xn_given_yn_age = np.zeros((2, 2, len(age_vec)))
    
    for curr_age in range(len(age_vec)):
        if curr_age == 0:
            ev_mat = np.eye(A.shape[0])
        else:
            ev_mat = np.linalg.matrix_power(A, curr_age)
        
        # Apply evolution matrix
        p_xn_given_yn_age[:, 0, curr_age] = p_xn_given_yn[0, :] @ ev_mat
        p_xn_given_yn_age[:, 1, curr_age] = p_xn_given_yn[1, :] @ ev_mat
    
    # Compute entropy conditioned on received value and on age
    h_given_yn_age = np.zeros((2, len(age_vec)))
    
    for curr_age in range(len(age_vec)):
        # H(X | Y=0, age)
        p0_y0 = p_xn_given_yn_age[0, 0, curr_age]
        p1_y0 = p_xn_given_yn_age[1, 0, curr_age]
        
        h_y0 = 0.0
        if p0_y0 > 0:
            h_y0 -= p0_y0 * np.log2(p0_y0)
        if p1_y0 > 0:
            h_y0 -= p1_y0 * np.log2(p1_y0)
        h_given_yn_age[0, curr_age] = h_y0
        
        # H(X | Y=1, age)
        p0_y1 = p_xn_given_yn_age[0, 1, curr_age]
        p1_y1 = p_xn_given_yn_age[1, 1, curr_age]
        
        h_y1 = 0.0
        if p0_y1 > 0:
            h_y1 -= p0_y1 * np.log2(p0_y1)
        if p1_y1 > 0:
            h_y1 -= p1_y1 * np.log2(p1_y1)
        h_given_yn_age[1, curr_age] = h_y1
    
    # Joint pmf p(yn, age)
    # Step 1: pmf of Age
    pmf_age = ps * (1 - ps) ** age_vec
    
    # Step 2: pmf of yn, p(yn)
    p_yn = np.zeros(2)
    p_yn[0] = np.sum(p_area * (pi0 * lambda_vec + pi1 * (1 - lambda_vec)))
    p_yn[1] = np.sum(p_area * (pi1 * lambda_vec + pi0 * (1 - lambda_vec)))
    
    # Overall joint distribution
    p_yn_age = np.zeros((2, len(age_vec)))
    p_yn_age[0, :] = p_yn[0] * pmf_age
    p_yn_age[1, :] = p_yn[1] * pmf_age
    
    # Average uncertainty
    H = 0.0
    H += np.sum(h_given_yn_age[0, :] * p_yn_age[0, :])
    H += np.sum(h_given_yn_age[1, :] * p_yn_age[1, :])
    
    # Compute the CDF
    h_stat = -pi0 * np.log2(pi0) - pi1 * np.log2(pi1)
    max_entropy = np.max(h_given_yn_age)
    gamma_vec = np.linspace(0, max_entropy, 1000)
    cdf_h = np.zeros(len(gamma_vec))
    
    for gg, curr_thr in enumerate(gamma_vec):
        # Find indices where entropy <= threshold
        mask = h_given_yn_age <= curr_thr
        cdf_h[gg] = np.sum(p_yn_age[mask])
    
    return H, cdf_h, gamma_vec


# Test function to validate the implementation
def test_stif_h_agnostic():
    """
    Test function with some reasonable parameters
    """
    # Example parameters
    R = 1.0
    K = 5
    alpha = 2.0
    rho = 1.0
    zeta = 0.1
    q = 0.1
    beta = 2.0
    
    H, cdf_h, gamma_vec = stif_h_agnostic_new(R, K, alpha, rho, zeta, q, beta)
    
    print(f"Average entropy H: {H:.6f}")
    print(f"CDF shape: {cdf_h.shape}")
    print(f"Gamma vector range: [{gamma_vec[0]:.6f}, {gamma_vec[-1]:.6f}]")
    
    return H, cdf_h, gamma_vec


if __name__ == "__main__":
    test_stif_h_agnostic()