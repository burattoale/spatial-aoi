"""
Location-aware Python translation of stif_H_agnostic_new.m
Computes spatial entropy H with H-agnostic approach for spatial-AoI systems
where the zone of origin is encoded in the received symbols.

Symbol encoding:
- 0, 1, ..., K-1: receiving 0 from zones 0, 1, ..., K-1 respectively
- K, K+1, ..., 2K-1: receiving 1 from zones 0, 1, ..., K-1 respectively
"""

import numpy as np
from typing import Tuple, List
from numba import jit


def stif_h_agnostic_loc_aware(R: float, K: int, alpha: float, rho: float, zeta: float, 
                              q: float, beta: float, epsilon: float = 0.0) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Location-aware STIF H-agnostic function
    
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
        Average uncertainty (joint entropy over symbol and location)
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
    
    # Initialize probability matrices for location-aware case
    # Now we have 2K possible received symbols (0..K-1 for receiving 0, K..2K-1 for receiving 1)
    num_symbols = 2 * K
    
    # p(symbol_received | x, age) where symbol encodes both value and zone
    # symbol_received: 0..K-1 (received 0 from zone 0..K-1), K..2K-1 (received 1 from zone 0..K-1)
    p_symbol_given_x_age = np.zeros((num_symbols, 2, len(age_vec)))
    
    # Step 1: Compute p(symbol | x) for each zone and symbol value
    # For location-aware case, we need to compute joint probabilities
    
    # p(receive symbol s | true value x)
    p_symbol_given_x = np.zeros((num_symbols, 2))
    
    for kk in range(K):
        # Symbols 0..K-1: receiving value 0 from zones 0..K-1
        symbol_0_from_zone_kk = kk
        # Symbols K..2K-1: receiving value 1 from zones 0..K-1  
        symbol_1_from_zone_kk = K + kk
        
        # p(receive 0 from zone kk | X=0) = p(area=kk) * p(success) * p(0|0,zone_kk)
        p_symbol_given_x[symbol_0_from_zone_kk, 0] = ps * p_area[kk] * lambda_vec[kk]
        
        # p(receive 0 from zone kk | X=1) = p(area=kk) * p(success) * p(0|1,zone_kk)
        p_symbol_given_x[symbol_0_from_zone_kk, 1] = ps * p_area[kk] * (1 - lambda_vec[kk])
        
        # p(receive 1 from zone kk | X=0) = p(area=kk) * p(success) * p(1|0,zone_kk)
        p_symbol_given_x[symbol_1_from_zone_kk, 0] = ps * p_area[kk] * (1 - lambda_vec[kk])
        
        # p(receive 1 from zone kk | X=1) = p(area=kk) * p(success) * p(1|1,zone_kk)
        p_symbol_given_x[symbol_1_from_zone_kk, 1] = ps * p_area[kk] * lambda_vec[kk]
    
    # Step 2: Compute p(x | symbol) using Bayes rule
    p_x_given_symbol = np.zeros((2, num_symbols))
    
    for s in range(num_symbols):
        # p(X | symbol=s)
        denominator = pi0 * p_symbol_given_x[s, 0] + pi1 * p_symbol_given_x[s, 1]
        if denominator > 0:
            p_x_given_symbol[0, s] = pi0 * p_symbol_given_x[s, 0] / denominator
            p_x_given_symbol[1, s] = pi1 * p_symbol_given_x[s, 1] / denominator
    
    # Step 3: Evolve over age using Markov chain
    p_x_given_symbol_age = np.zeros((2, num_symbols, len(age_vec)))
    
    for curr_age in range(len(age_vec)):
        if curr_age == 0:
            ev_mat = np.eye(A.shape[0])
        else:
            ev_mat = np.linalg.matrix_power(A, curr_age)
        
        # Apply evolution matrix
        for s in range(num_symbols):
            p_x_given_symbol_age[:, s, curr_age] = p_x_given_symbol[:, s] @ ev_mat
    
    # Step 4: Compute entropy conditioned on received symbol and age
    # This is the joint entropy H(X | symbol, age)
    h_given_symbol_age = np.zeros((num_symbols, len(age_vec)))
    
    for curr_age in range(len(age_vec)):
        for s in range(num_symbols):
            # H(X | symbol=s, age)
            p0 = p_x_given_symbol_age[0, s, curr_age]
            p1 = p_x_given_symbol_age[1, s, curr_age]
            
            h_val = 0.0
            if p0 > 0:
                h_val -= p0 * np.log2(p0)
            if p1 > 0:
                h_val -= p1 * np.log2(p1)
            h_given_symbol_age[s, curr_age] = h_val
    
    # Step 5: Compute joint distribution p(symbol, age)
    # First, compute p(symbol) by marginalizing over source symbols in stationary conditions
    p_symbol = np.zeros(num_symbols)
    
    for s in range(num_symbols):
        p_symbol[s] = pi0 * p_symbol_given_x[s, 0] + pi1 * p_symbol_given_x[s, 1]
    
    # Age distribution
    pmf_age = ps * (1 - ps) ** age_vec
    
    # Joint distribution p(symbol, age) - assuming independence between symbol reception and age
    p_symbol_age = np.zeros((num_symbols, len(age_vec)))
    for s in range(num_symbols):
        p_symbol_age[s, :] = p_symbol[s] * pmf_age
    
    # Step 6: Compute average uncertainty (joint entropy)
    H = 0.0
    for s in range(num_symbols):
        H += np.sum(h_given_symbol_age[s, :] * p_symbol_age[s, :])
    
    # Step 7: Compute the CDF
    max_entropy = np.max(h_given_symbol_age)
    gamma_vec = np.linspace(0, max_entropy, 1000)
    cdf_h = np.zeros(len(gamma_vec))
    
    for gg, curr_thr in enumerate(gamma_vec):
        # Find indices where entropy <= threshold
        mask = h_given_symbol_age <= curr_thr
        cdf_h[gg] = np.sum(p_symbol_age[mask])
    
    return H, cdf_h, gamma_vec


def compare_location_aware_vs_standard(R: float = 1.0, K: int = 5, alpha: float = 2.0, 
                                       rho: float = 1.0, zeta: float = 0.1, 
                                       q: float = 0.1, beta: float = 2.0) -> None:
    """
    Compare location-aware vs standard STIF implementations
    """
    try:
        from stif_h_agnostic_python import stif_h_agnostic_new
    except ImportError:
        print("Could not import standard STIF function for comparison")
        return
    
    print("Comparing Location-Aware vs Standard STIF H-agnostic:")
    print(f"Parameters: R={R}, K={K}, α={alpha}, ρ={rho}, ζ={zeta}, q={q}, β={beta}")
    print("-" * 60)
    
    # Standard version
    H_std, cdf_std, gamma_std = stif_h_agnostic_new(R, K, alpha, rho, zeta, q, beta)
    
    # Location-aware version
    H_loc, cdf_loc, gamma_loc = stif_h_agnostic_loc_aware(R, K, alpha, rho, zeta, q, beta)
    
    print(f"Standard H-agnostic entropy:      {H_std:.6f}")
    print(f"Location-aware H-agnostic entropy: {H_loc:.6f}")
    print(f"Difference (loc_aware - standard): {H_loc - H_std:.6f}")
    print(f"Relative difference:               {((H_loc - H_std) / H_std * 100):+.2f}%")
    
    # Additional analysis
    print(f"\nSymbol space comparison:")
    print(f"Standard version: 2 symbols (0, 1)")
    print(f"Location-aware:   {2*K} symbols (0..{K-1} for value 0, {K}..{2*K-1} for value 1)")
    
    return H_std, H_loc


# Test function to validate the implementation
def test_stif_h_agnostic_loc_aware():
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
    
    H, cdf_h, gamma_vec = stif_h_agnostic_loc_aware(R, K, alpha, rho, zeta, q, beta)
    
    print(f"Location-aware average entropy H: {H:.6f}")
    print(f"CDF shape: {cdf_h.shape}")
    print(f"Gamma vector range: [{gamma_vec[0]:.6f}, {gamma_vec[-1]:.6f}]")
    print(f"Number of symbols: {2*K} (0..{K-1} for value 0, {K}..{2*K-1} for value 1)")
    
    return H, cdf_h, gamma_vec


if __name__ == "__main__":
    test_stif_h_agnostic_loc_aware()
    print("\n" + "="*60)
    compare_location_aware_vs_standard()