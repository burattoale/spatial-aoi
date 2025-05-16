import numpy as np
import math
from numba import jit
from matplotlib import pyplot as plt

class SourceMM(object):
    """
    Contains the binary markov process state of the source
    """

    def __init__(self, q:float, eta:float):
        self._A = np.array([[1 - q, q],
                            [eta * q, 1 - eta * q]])
        self._steady_state = self._compute_steady_state()
        self.state_names = [0, 1]
        self.current_state = 0
        self.rng = np.random.default_rng()

    @property
    def A(self):
        return self._A
    
    @property
    def pi(self):
        return self._steady_state
    
    @property
    def state(self):
        return str(self.current_state)
    
    def step(self):
        next_state = self.rng.choice(self.state_names, p=self.A[self.current_state])
        self.current_state = next_state
        return str(self.current_state)

    def _compute_steady_state(self):
        dim = self.A.shape[0]
        q = (self.A-np.eye(dim))
        ones = np.ones(dim)
        q = np.c_[q,ones]
        QTQ = np.dot(q, q.T)
        bQT = np.ones(dim)
        return np.linalg.solve(QTQ,bQT)
    

@jit
def lam(d:int, alpha:float=0.02, R:float=10):
    return 1 #1 / (1 + d * R)**alpha

@jit   
def cond_prob(y, x, zeta, epsilon, m:int, K:int, alpha:float=0.02, R:float=10):
    """
    Compute the conditional probability of Y=y|X=x
    """
    ps = m * zeta * (1 - epsilon) * (1 - zeta * (1 - epsilon))**(m-1)
    pi = (1 - zeta * (1 - epsilon))**m
    if y == "0" or y == "1":
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
    if y == "C":
        return 1 - ps - pi
    if y == "I":
        return pi
    raise NotImplementedError

@jit
def sequence_entropy(lam:float):
    out = 0
    for x in [0, 1]:
        out += np.exp(-x * lam) / (1 + np.exp(-lam)) + np.log2((1 + np.exp(-lam)) / (np.exp(-x *lam)))
    return out



if __name__ == "__main__":
    sim_length = 10
    # sim parameters
    q = 5e-3
    eta = 5
    zeta = 1e-4
    epsilon = 0.1
    rho = 5e-2
    R = 10
    K = 1
    m = 1#math.floor(rho * np.pi * R*K)
    print(f"Number of sensors: {m}")
    alpha = 0.02

    hmm = SourceMM(q, eta) # process markov chain
    x_symbols = ["0", "1"]
    y_symbols = ["0", "1", "C", "I"]
    rng = np.random.default_rng()
    # sbagliato!!! Il processo Y scritto così è slegato dal processo hidden
    Y = rng.choice(y_symbols, size=sim_length)

    cond_prob0 = cond_prob(Y[0], "0", zeta, epsilon, m, K, alpha)
    cond_prob1 = cond_prob(Y[0], "1", zeta, epsilon, m, K, alpha)
    alpha_vec = np.empty(2)
    alpha_vec[0] = hmm.pi[0] * cond_prob0
    alpha_vec[1] = hmm.pi[1] * cond_prob1

    ###### Main simulation loop ######
    # for i in range(1, sim_length):
    #     temp0, temp1 = 0, 0
    #     for id in range(2): # only two states
    #         temp0 += alpha_vec[id] * hmm.A[id, 0]
    #         temp1 += alpha_vec[id] * hmm.A[id, 1]
    #     alpha_vec[0] = temp0 * cond_prob(Y[i], "0", zeta, epsilon, m, K, alpha)
    #     alpha_vec[1] = temp1 * cond_prob(Y[i], "1", zeta, epsilon, m, K, alpha)
# 
    # lambda_n = np.log(alpha_vec[0]/alpha_vec[1])
    # print(lambda_n)

    ###### Alternative formulation for lambda_n ######
    try:
        lambda_other = np.log(cond_prob(Y[0], "0", zeta, epsilon, m, K, alpha) / cond_prob(Y[0], "1", zeta, epsilon, m, K, alpha)) + np.log(hmm.A[0,0]/hmm.A[0,1])
    except:
        lambda_other = np.log(hmm.A[0,0]/hmm.A[0,1])
    entropies = np.zeros(sim_length)
    for i in range (1, sim_length):
        print(Y[i])
        num = cond_prob(Y[i], "0", zeta, epsilon, m, K, alpha)
        den = cond_prob(Y[i], "1", zeta, epsilon, m, K, alpha)
        try:
            first_part = np.log(num/den)
        except:
            first_part=0
        lambda_other = first_part + np.log((hmm.A[0,0] + hmm.A[1,0] * np.exp(-lambda_other)) / (hmm.A[0,1] + hmm.A[1,1] * np.exp(-lambda_other)))
        print(lambda_other)
        entropies[i] = sequence_entropy(lambda_other)

    # print(f"Entropy alpha method: {sequence_entropy(lambda_n)}")
    print(f"Entropy Liva method: {sequence_entropy(lambda_other)}")

    plt.figure()
    source_entropy = - hmm.pi[0] * np.log(hmm.pi[0]) - hmm.pi[1] * np.log(hmm.pi[1])
    plt.plot(np.arange(sim_length), entropies, label=r"$H(X_n \mid Y^n=y^n)$")
    plt.plot(np.arange(sim_length), np.ones(sim_length) * source_entropy, linestyle="-.", label="$H(X)$")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/sequence_entropy.png", bbox_inches="tight", pad_inches=0)
    plt.show()



