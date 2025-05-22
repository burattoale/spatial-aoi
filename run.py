from environment import NodeDistribution
from poibin import PoiBin
from numba import jit
import math
import numpy as np

class MM(object):
    def __init__(self, q, eta):
        self._A = np.array([[1-q, q], [eta*q, 1-eta*q]])
        self._pi = self._compute_steady_state()
        self.rng = np.random.default_rng()
        self._state = self.rng.choice([0, 1], p=self._pi)

    @property
    def A(self):
        return self._A
    
    @property
    def pi(self):
        return self._pi
    
    @property
    def state(self):
        return self._state
    
    def step(self):
        next_state = self.rng.choice([0, 1], p=self.A[int(self.state)])        
        self._state = next_state
        return next_state

    def _compute_steady_state(self):
        dim = self._A.shape[0]
        q = (self._A-np.eye(dim))
        ones = np.ones(dim)
        q = np.c_[q,ones]
        QTQ = np.dot(q, q.T)
        bQT = np.ones(dim)
        return np.linalg.solve(QTQ,bQT)

@jit
def prob_y_given_x_0(x:int, y:int, K:int, R:int, alpha:float):
    out = 0
    for d in range(K):
        if y == x:
            out += (2*d + 1) / (K**2 * (1 + d * R)**alpha)
        else:
            out += (2*d + 1) / (K**2) * (1 - 1 / (1 + d * R)**alpha)
    return out

def prob_x_given_y_0(x:int, y:int, mm:MM, K:int, R:int, alpha:float):
    p_y_given_x_0 = prob_y_given_x_0(x, y, K, R, alpha)
    return p_y_given_x_0 * mm.pi[x] / (prob_y_given_x_0(0, y, K, R, alpha) * mm.pi[0] + prob_y_given_x_0(1, y, K, R, alpha) * mm.pi[1])

def prob_x_given_y_delta(p_x_given_y_0:np.ndarray, mm:MM, delta:int):
    return p_x_given_y_0.T @ np.linalg.matrix_power(mm.A, delta)

def h_y_delta(p_x_given_y_0:np.ndarray, mm:MM, delta:int, x_symbols=[0,1]):
    p_x_given_y_delta = prob_x_given_y_delta(p_x_given_y_0, mm, delta)
    out = 0
    for x in x_symbols:
        out -= p_x_given_y_delta[x] * np.log2(p_x_given_y_delta[x])
    return out

def p_y(y:int, mm, K, R, alpha):
    prob_y_given_0_0 = prob_y_given_x_0(0, y, K, R, alpha)
    prob_y_given_1_0 = prob_y_given_x_0(1, y, K, R, alpha)
    return prob_y_given_0_0 * mm.pi[0] + prob_y_given_1_0 * mm.pi[1]

def ps(m:int, zeta:float, epsilon):
    dist = PoiBin([zeta*(1-epsilon)]*m)
    return dist.pmf(1)

def p_delta(ps:float, delta:int):
    return ps * (1-ps)**(delta)

def entropy(mm, K, R, alpha, m, zeta, epsilon, states):
    sum = 0
    for state in states:
        y, delta = state
        p_x_given_y_0 = np.array([prob_x_given_y_0(0, y, mm, K, R, alpha), prob_x_given_y_0(1, y, mm, K, R, alpha)])
        sum += h_y_delta(p_x_given_y_0, mm, delta) * p_y(y, mm, K, R, alpha) * p_delta(ps(m, zeta, epsilon), delta)
    return sum

if __name__ == "__main__":
    # PARAMETERS
    q = 5e-3
    eta = 1
    zeta = 1e-4
    epsilon = 0.1
    rho = 5e-2
    alpha = 0.02
    R = 10
    x_symbols = [0, 1]
    y_symbols = [0, 1]

    #### SIM Length ####
    sim_length = 1000

    mm = MM(q, eta)

    entropies = []
    without_tx = []
    ####################
    #### SIMULATION ####
    ####################
    for K in range(4,60):
        m = math.floor(rho * np.pi * R*K)
        p_succ = ps(m, zeta, epsilon)
        rng = np.random.default_rng()
        last_y, delta = 0, 0 # initialize variables
        sim_entropies = []
        for _ in range(20):
            states = []
            for i in range(sim_length):
                if i == 0:
                    x_state = mm.state
                    last_y = x_state
                    delta = 0
                    states.append((last_y, delta))
                    continue
                x_state = mm.step()
                if rng.random() <= p_succ:
                    last_y = x_state
                    delta = 0
                else:
                    delta += 1
                states.append((last_y, delta))
            states_entropy = list(set(states))
            sim_entropies.append(entropy(mm, K, R, alpha, m, zeta, epsilon, states_entropy))
        entropies.append(np.mean(sim_entropies))


    import matplotlib.pyplot as plt

    source_entropy = np.sum(- mm.pi * np.log2(mm.pi))
    for state in states:
        y, delta = state
        p_x_given_y_0 = np.array([prob_x_given_y_0(0, y, mm, K, R, alpha), prob_x_given_y_0(1, y, mm, K, R, alpha)])
        without_tx.append(h_y_delta(p_x_given_y_0, mm, delta))

    plt.plot(entropies)
    plt.xticks(np.arange(0,57, 4),np.arange(40, 601, step=40))
    plt.show()

    plt.figure()
    plt.plot(without_tx)
    plt.plot([source_entropy]*len(without_tx))
    plt.grid(True)
    plt.show()