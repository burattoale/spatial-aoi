import numpy as np
from numba import jit

@jit
def lam(d:int, alpha:float=0.02, R:float=10, bypass=False):
    if bypass:
        return 1
    return 1 / (1 + d * R)**alpha


def cond_prob_y_given_x(y, x, zeta, epsilon, m:int, K:int, lambda_mat:np.ndarray, tx_prob_per_bucket:np.ndarray, p_d_vector:np.ndarray, x_states_cardinality:int, poibin:bool=False, p_succ=None):
    """
    Compute the conditional probability of Y=y|X=x

    """
    if poibin:
        assert p_succ is not None
        ps = p_succ
    else:
        ps = m * zeta * (1 - epsilon) * (1 - zeta * (1 - epsilon))**(m-1)
    
    sum_prob_buckets = np.sum(tx_prob_per_bucket * p_d_vector)
    if y < x_states_cardinality:
        total = 0
        for d in range(K):
            weight = p_d_vector[d] * tx_prob_per_bucket[d]/sum_prob_buckets
            total += lambda_mat[x,y,d] * weight
        total *= ps
        return total
    if y == x_states_cardinality:
        return 1 - ps
    raise ValueError(f"The value y = {y} is not valid for building a non spatial-aware model.")

@jit   
def cond_prob_y_given_x_spatial(y, x, zeta, epsilon, m:int, K:int, lambda_mat:np.ndarray, tx_prob_per_bucket:np.ndarray, p_d_vector:np.ndarray, x_states_cardinality:int, poibin:bool=False, p_succ=None):
    """
    Compute the conditional probability of Y=y|X=x considering the origin of the message
    """
    if poibin:
        assert p_succ is not None
        ps = p_succ
    else:
        ps = m * zeta * (1 - epsilon) * (1 - zeta * (1 - epsilon))**(m-1)

    valued_symbol = y // K
    d_idx = y % K
    sum_prob_buckets = np.sum(tx_prob_per_bucket * p_d_vector)
    if valued_symbol < x_states_cardinality:
        weight = p_d_vector[d_idx] * tx_prob_per_bucket[d_idx]/sum_prob_buckets 
        total = ps * lambda_mat[x, valued_symbol, d_idx] * weight
        return total
    if valued_symbol >= x_states_cardinality:
        return 1 - ps
    raise ValueError(f"The value y = {y} is not valid for building a non spatial-aware model.")


@jit
def sequence_entropy(lam:float):
    out = 0
    for x in [0, 1]:
        out += np.exp(-x * lam) / (1 + np.exp(-lam)) + np.log2((1 + np.exp(-lam)) / (np.exp(-x *lam)))
    return out