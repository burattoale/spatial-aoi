# forgetful_receiver_funcs.py
import numpy as np
from numba import jit
from typing import List
import os
import pickle

# Assuming poibin is installed: pip install poibin
from .poibin import PoiBin
from .hmm_funcs import lam
from .matrix_builders import generate_lambda_matrix

@jit 
def prob_y_given_x_0(x:int, y:int, K:int, tx_prob_per_bucket:np.ndarray, lambda_mat:np.ndarray, p_d_vector:np.ndarray): # Renamed R to R_unit for clarity
    out = 0
    sum_prob_buckets = np.sum(tx_prob_per_bucket * p_d_vector)
    for d_idx in range(K): # d is bucket index
        
        weight = p_d_vector[d_idx] * tx_prob_per_bucket[d_idx]/sum_prob_buckets

        out += weight * lambda_mat[x,y,d_idx]
    return out

@jit
def prob_y_given_x_0_spatial(x:int, y:int, K:int, p_succ:float, prob_d_given_tx_vector:np.ndarray, lambda_mat:np.ndarray):
    valued_symbol = y // K
    d_idx = y % K
    if valued_symbol == 0 or valued_symbol == 1:
        if x == valued_symbol:
            total = p_succ * prob_d_given_tx_vector[d_idx] * lambda_mat[d_idx]
        else:
            total = p_succ * prob_d_given_tx_vector[d_idx] * (1-lambda_mat[d_idx])
        return total
    elif valued_symbol == 2:
        return 1 - p_succ
    else:
        raise NotImplementedError

@jit
def prob_x_given_y_0(x:int, y:int, pi:np.ndarray, K:int, tx_prob_per_bucket:np.ndarray, lambda_mat:np.ndarray, p_d_vector:np.ndarray):
    # P(X=x | Y=y, Delta=0) = P(Y=y | X=x, Delta=0) * P(X=x) / P(Y=y | Delta=0)
    # P(Y=y | Delta=0) = sum_x' P(Y=y | X=x', Delta=0) * P(X=x')
    p_y_given_x_0_val = prob_y_given_x_0(x, y, K, tx_prob_per_bucket, lambda_mat, p_d_vector)
    
    p_y_given_0_0 = prob_y_given_x_0(0, y, K, tx_prob_per_bucket, lambda_mat, p_d_vector)
    p_y_given_1_0 = prob_y_given_x_0(1, y, K, tx_prob_per_bucket, lambda_mat, p_d_vector)
    
    p_y_0 = p_y_given_0_0 * pi[0] + p_y_given_1_0 * pi[1]
    
    if p_y_0 == 0: # Avoid division by zero; implies this y is impossible
        return 0.0 
    return p_y_given_x_0_val * pi[x] / p_y_0

@jit
def prob_x_given_y_0_spatial(x:int, y:int, pi:np.ndarray, K:int, p_succ:float, prob_d_given_tx_vector:np.ndarray, lambda_mat:np.ndarray):
    # P(X=x | Y=y, Delta=0) = P(Y=y | X=x, Delta=0) * P(X=x) / P(Y=y | Delta=0)
    # P(Y=y | Delta=0) = sum_x' P(Y=y | X=x', Delta=0) * P(X=x')
    p_y_given_x_0_val = prob_y_given_x_0_spatial(x, y, K, p_succ, prob_d_given_tx_vector, lambda_mat)
    
    p_y_given_0_0 = prob_y_given_x_0_spatial(0, y, K, p_succ, prob_d_given_tx_vector, lambda_mat)
    p_y_given_1_0 = prob_y_given_x_0_spatial(1, y, K, p_succ, prob_d_given_tx_vector, lambda_mat)
    
    p_y_0 = p_y_given_0_0 * pi[0] + p_y_given_1_0 * pi[1]
    
    if p_y_0 == 0: # Avoid division by zero; implies this y is impossible
        return 0.0 
    return p_y_given_x_0_val * pi[x] / p_y_0

@jit
def prob_x_given_y_delta(p_x_given_y_0_vec:np.ndarray, A_delta_power:np.ndarray):
    # p_x_given_y_0_vec is a row vector [P(X_0=0|Y_0), P(X_0=1|Y_0)]
    # We want P(X_delta | Y_0) = P(X_0 | Y_0) @ A^delta
    # Result is a row vector [P(X_delta=0|Y_0), P(X_delta=1|Y_0)]
    # Ensure p_x_given_y_0_vec is a 1D array for consistent matrix multiplication result
    return np.dot(p_x_given_y_0_vec.flatten(), A_delta_power)

@jit
def h_y_delta(p_x_given_y_0_vec:np.ndarray, A_delta_power:np.ndarray, x_symbols=None):
    # Calculates H(X_delta | Y_0)
    if x_symbols is None:
        x_symbols = np.array([0, 1]) # Assuming binary source
    
    p_x_delta_given_y_0 = prob_x_given_y_delta(p_x_given_y_0_vec, A_delta_power)
    
    out = 0
    for i in range(len(x_symbols)): # Iterate over possible states of X_delta
        p_val = p_x_delta_given_y_0[i]
        if p_val > 0: # Avoid log(0)
            out -= p_val * np.log2(p_val)
    return out

@jit
def p_y(y:int, pi:np.ndarray, K:int, tx_prob_per_bucket:np.ndarray, lambda_mat:np.ndarray, p_d_vector:np.ndarray):
    # P(Y=y | Delta=0) (marginal probability of observing y when an update occurs)
    prob_y_given_0_0 = prob_y_given_x_0(0, y, K, tx_prob_per_bucket, lambda_mat, p_d_vector)
    prob_y_given_1_0 = prob_y_given_x_0(1, y, K, tx_prob_per_bucket, lambda_mat, p_d_vector)
    return prob_y_given_0_0 * pi[0] + prob_y_given_1_0 * pi[1]

def ps_calc(m:int, zeta:float|List, epsilon:float): # Renamed to ps_calc to avoid conflict if ps is a variable
    """
    Probability of the receiver getting an update from the m nodes.
    The original used PoiBin(...).pmf(1) for exactly one success.
    It's more standard that an update occurs if *at least one* node succeeds.
    P(at least one success) = 1 - P(all fail).
    P(one node attempts and succeeds) = zeta * (1-epsilon).
    P(one node fails to provide update) = 1 - zeta * (1-epsilon).
    P(all m nodes fail) = (1 - zeta * (1-epsilon))^m
    So, P(at least one success) = 1 - (1 - zeta * (1-epsilon))^m
    
    However, sticking to the provided function's PoiBin use:
    This means p_i = zeta * (1-epsilon) for each node.
    We need a list of these probabilities for PoiBin.
    """
    if isinstance(zeta, list):
        zetas = np.array(zeta)
        assert len(zetas) == m
    else:
        zetas = np.array([zeta] * m)
    if m == 0: return 0.0 # No nodes, no updates
    prob_one_node_success = zetas * (1 - epsilon)
    prob_one_node_success = np.clip(prob_one_node_success, 0 ,1)

    # Using the provided PoiBin logic for pmf(1)
    dist = PoiBin(prob_one_node_success)
    return dist.pmf(1) # Probability of *exactly one* success


@jit
def p_delta_calc(prob_success_update:float, delta:int): # Renamed to p_delta_calc
    # P(Delta = delta) = P(update) * P(no update)^delta
    # This is probability of current age being delta, given an update just happened (delta=0)
    # or delta failures followed by a success.
    if delta < 0: return 0.0
    return prob_success_update * (1 - prob_success_update)**delta

def overall_entropy(A:np.ndarray, 
                    pi:np.ndarray, 
                    K:int, 
                    m:int, 
                    zeta:float|List[float], 
                    epsilon:float,
                    alpha:float,
                    R_unit:float,
                    X_symbols:List[int],
                    Y_symbols:List[int],
                    tx_prob_per_bucket: List|np.ndarray, 
                    max_delta_considered: int = 20):
    """
    Calculates the theoretical average entropy H(X_t | Y_n, Delta_n).
    This is E[h_y_delta] = sum_{y,delta} h_y_delta(..) * P(Y=y, Delta=delta)
    P(Y=y, Delta=delta) = P(Y=y | Delta=0) * P(Delta=delta)
                         = p_y(y,...) * p_delta(ps_val, delta)
    The 'states' argument in the original function seems to imply iterating over (y,delta) pairs.
    """
    total_entropy_sum = 0
    
    # Probability of receiver getting an update (at least one node succeeds)
    prob_update_success = ps_calc(m, zeta, epsilon)
    if prob_update_success == 0.0 and m > 0 : # Edge case if zeta or (1-epsilon) is 0
        print("Warning: prob_update_success is 0. Can't receive messages.")
        # If no updates, AoI grows indefinitely, entropy might approach H(X)
        # This theoretical calculation might break down.
        # For simulation, this means AoI will keep increasing.
        # For theoretical, P(Delta=delta) is ill-defined if ps_calc = 0 or 1.
        # If ps_calc = 0, p_delta is 0 for delta=0, and ill-defined for delta > 0.
        # If ps_calc = 1, p_delta is 1 for delta=0, and 0 for delta > 0.

    filename = 'matrix_power_cache.pkl'

    if os.path.exists(filename):
        # Load the cache from the pickle file
        with open(filename, 'rb') as f:
            matrix_power_cache = pickle.load(f)
    else:
        matrix_power_cache = {}
        matrix_power_cache[0] = np.eye(A.shape[0])
        matrix_power_cache[1] = A.copy()

    # lambda_mat
    lambda_mat = generate_lambda_matrix(len(X_symbols), K, alpha, R_unit)
    p_d_vector = [(2 * d + 1) / (K**2) for d in range(K)]
    p_d_vector = np.array(p_d_vector)

    for y_val in Y_symbols:
        # P(X_0 | Y_0=y_val)
        p_x_g_y0_vec = np.array([
            prob_x_given_y_0(x, y_val, pi, K, tx_prob_per_bucket, lambda_mat, p_d_vector) for x in X_symbols 
        ])

        # Marginal probability of receiving y_val, given an update occurred
        prob_y_val_at_update = p_y(y_val, pi, K, tx_prob_per_bucket, lambda_mat, p_d_vector)

        for delta_val in range(max_delta_considered + 1): # Sum over a practical range of delta
            # Conditional entropy H(X_delta | Y_0=y_val)
            if delta_val not in matrix_power_cache:
                matrix_power_cache[delta_val] = np.linalg.matrix_power(A, delta_val)
            A_delta_power = matrix_power_cache[delta_val]
            h_val = h_y_delta(p_x_g_y0_vec, A_delta_power)
            
            # Probability of this (y_val, delta_val) situation occurring
            # P(Delta = delta_val)
            prob_delta = p_delta_calc(prob_update_success, delta_val)
            
            # P(Y_n=y_val, Delta_n=delta_val) = P(Y_0=y_val) * P(Delta=delta_val)
            # (Assuming Y_n when Delta_n=d is statistically same as Y_0 when Delta_0=0)
            joint_prob_y_delta = prob_y_val_at_update * prob_delta
            
            total_entropy_sum += h_val * joint_prob_y_delta

    # save the dictionary if it is new
    if not os.path.exists(filename):
        with open(filename, 'wb') as f:
            pickle.dump(matrix_power_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    # Normalize if p_delta does not sum to 1 over max_delta_considered
    # sum_prob_delta = sum(p_delta_calc(prob_update_success, d) for d in range(max_delta_considered + 1))
    # if sum_prob_delta > 0 and not np.isclose(sum_prob_delta, 1.0):
    #     total_entropy_sum /= sum_prob_delta # Normalize

    return total_entropy_sum