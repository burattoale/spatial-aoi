import numpy as np
from numba import jit

@jit
def lam(d:int, alpha:float=0.02, R:float=10, bypass=False):
    if bypass:
        return 1
    return 1 / (1 + d * R)**alpha


def cond_prob_y_given_x(y, x, zeta, epsilon, m:int, K:int, alpha:float=0.02, R:float=10, poibin:bool=False, p_succ=None, p_idle=None):
    """
    Compute the conditional probability of Y=y|X=x
    """
    if poibin:
        assert p_succ is not None and p_idle is not None
        ps = p_succ
        pi = p_idle
    else:
        ps = m * zeta * (1 - epsilon) * (1 - zeta * (1 - epsilon))**(m-1)
        pi = (1 - zeta * (1 - epsilon))**m
    if y == 0 or y == 1:
        if x == y:
            total = 0
            for d in range(K):
                total += lam(d, alpha, R) * (2*d + 1) / K**2
            total *= ps
        else:
            total = 0
            for d in range(K):
                total += (1-lam(d, alpha, R)) * (2*d + 1) / K**2
            total *= ps
        return total
    if y == 2:
        return 1 - ps - pi
    if y == 3:
        return pi
    raise NotImplementedError

def cond_prob_y_given_x_non_binary(y, x, zeta, epsilon, m:int, K:int, lambda_mat:np.ndarray, states_cardinality:int, poibin:bool=False, p_succ=None):
    if poibin:
        assert p_succ is not None
        ps = p_succ
    else:
        ps = m * zeta * (1 - epsilon) * (1 - zeta * (1 - epsilon))**(m-1)
    if y < states_cardinality:    
        total = 0
        for d in range(K):
            total += lambda_mat[x,y,d] * (2*d + 1) / K**2
        total *= ps
        return total
    else: 
        return 1 - ps

@jit   
def cond_prob_y_given_x_spatial(y, x, zeta, epsilon, m:int, K:int, alpha:float=0.02, R:float=10):
    """
    Compute the conditional probability of Y=y|X=x considering the origin of the message
    """
    ps = m * zeta * (1 - epsilon) * (1 - zeta * (1 - epsilon))**(m-1)
    pi = (1 - zeta * (1 - epsilon))**m
    valued_symbol = y // K
    if valued_symbol == 0 or valued_symbol == 1:
        if x == valued_symbol:
            total = ps * (2*(y%K) + 1) / K**2 * lam(y%K, alpha, R)
        else:
            total = ps * (2*(y%K) + 1) / K**2 * (1-lam(y%K, alpha, R))
        return total
    if valued_symbol == 2 and y % K == 0:
        return 1 - ps - pi
    if valued_symbol == 2 and y % K == 1 or valued_symbol == 3:
        return pi
    raise NotImplementedError


@jit
def sequence_entropy(lam:float):
    out = 0
    for x in [0, 1]:
        out += np.exp(-x * lam) / (1 + np.exp(-lam)) + np.log2((1 + np.exp(-lam)) / (np.exp(-x *lam)))
    return out