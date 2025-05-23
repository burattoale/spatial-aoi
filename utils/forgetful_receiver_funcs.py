import numpy as np
from numba import jit

from .poibin import PoiBin

@jit
def prob_y_given_x_0(x:int, y:int, K:int, R:int, alpha:float):
    out = 0
    for d in range(K):
        if y == x:
            out += (2*d + 1) / (K**2 * (1 + d * R)**alpha)
        else:
            out += (2*d + 1) / (K**2) * (1 - 1 / (1 + d * R)**alpha)
    return out

@jit
def prob_x_given_y_0(x:int, y:int, pi:np.ndarray, K:int, R:int, alpha:float):
    p_y_given_x_0 = prob_y_given_x_0(x, y, K, R, alpha)
    return p_y_given_x_0 * pi[x] / (prob_y_given_x_0(0, y, K, R, alpha) * pi[0] + prob_y_given_x_0(1, y, K, R, alpha) * pi[1])

@jit
def prob_x_given_y_delta(p_x_given_y_0:np.ndarray, A:np.ndarray, delta:int):
    return p_x_given_y_0.T @ np.linalg.matrix_power(A, delta)

@jit
def h_y_delta(p_x_given_y_0:np.ndarray, A:np.ndarray, delta:int, x_symbols=None):
    if x_symbols is None:
        x_symbols = np.array([0, 1])
    p_x_given_y_delta = prob_x_given_y_delta(p_x_given_y_0, A, delta)
    out = 0
    for x in x_symbols:
        out -= p_x_given_y_delta[x] * np.log2(p_x_given_y_delta[x])
    return out

@jit
def p_y(y:int, pi:np.ndarray, K, R, alpha):
    prob_y_given_0_0 = prob_y_given_x_0(0, y, K, R, alpha)
    prob_y_given_1_0 = prob_y_given_x_0(1, y, K, R, alpha)
    return prob_y_given_0_0 * pi[0] + prob_y_given_1_0 * pi[1]

def ps(m:int, zeta:float, epsilon):
    dist = PoiBin([zeta*(1-epsilon)]*m)
    return dist.pmf(1)

@jit
def p_delta(ps:float, delta:int):
    return ps * (1-ps)**(delta)

def entropy(A:np.ndarray, pi:np.ndarray, K, R, alpha, m, zeta, epsilon, states):
    sum = 0
    for state in states:
        y, delta = state
        p_succ = ps(m, zeta, epsilon)
        p_x_given_y_0 = np.array([prob_x_given_y_0(0, y, pi, K, R, alpha), prob_x_given_y_0(1, y, pi, K, R, alpha)])
        sum += h_y_delta(p_x_given_y_0, A, delta) * p_y(y, pi, K, R, alpha) * p_delta(p_succ, delta)
    return sum