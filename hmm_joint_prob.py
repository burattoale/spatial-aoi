import numpy as np
from matplotlib import pyplot as plt

from environment import HiddenMM
from utils import SimulationParameters
from utils import cond_prob, sequence_entropy


if __name__ == "__main__":
    sim_length = 1000
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
    x_symbols = ["0", "1"]
    y_symbols = ["0", "1", "C", "I"]

    params = SimulationParameters(q, eta, zeta, epsilon, m, K, alpha, R, x_symbols, y_symbols)

    hmm = HiddenMM(params) # process markov chain
    rng = np.random.default_rng()

    Y = np.empty(sim_length, dtype=object)
    Y[0] = hmm.state

    cond_prob0 = cond_prob(Y[0], "0", zeta, epsilon, m, K, alpha)
    cond_prob1 = cond_prob(Y[0], "1", zeta, epsilon, m, K, alpha)
    alpha_vec = np.empty(2)
    alpha_vec[0] = hmm.pi[0] * cond_prob0
    alpha_vec[1] = hmm.pi[1] * cond_prob1
    cs = np.zeros(sim_length)  # scaling
    cs[0] = np.sum(alpha_vec)
    alpha_vec /= cs[0]

    ###### Alternative formulation for lambda_n ######
    try:
        lambda_other = np.log(cond_prob(Y[0], "0", zeta, epsilon, m, K, alpha) / cond_prob(Y[0], "1", zeta, epsilon, m, K, alpha)) + np.log(hmm.A[0,0]/hmm.A[0,1])
    except ZeroDivisionError:
        lambda_other = np.log(hmm.A[0,0]/hmm.A[0,1])
    entropies = np.zeros(sim_length)
    
    for i in range (1, sim_length):
        Y[i] = hmm.step()
        num = cond_prob(Y[i], "0", zeta, epsilon, m, K, alpha)
        den = cond_prob(Y[i], "1", zeta, epsilon, m, K, alpha)
        try:
            first_part = np.log(num/den)
        except ZeroDivisionError:
            first_part = 0
        lambda_other = first_part + np.log((hmm.A[0,0] + hmm.A[1,0] * np.exp(-lambda_other)) / (hmm.A[0,1] + hmm.A[1,1] * np.exp(-lambda_other)))
        temp0, temp1 = 0, 0
        for id in range(2): # only two states
            temp0 += alpha_vec[id] * hmm.A[id, 0]
            temp1 += alpha_vec[id] * hmm.A[id, 1]
        alpha_vec[0] = temp0 * cond_prob(Y[i], "0", zeta, epsilon, m, K, alpha)
        alpha_vec[1] = temp1 * cond_prob(Y[i], "1", zeta, epsilon, m, K, alpha)
        cs[i] = np.sum(alpha_vec)
        alpha_vec /= cs[i]
        entropies[i] = sequence_entropy(lambda_other)

    lambda_n = np.log(alpha_vec[0]/alpha_vec[1])
    print(f"Entropy alpha method: {sequence_entropy(lambda_n)}")
    print(f"Entropy Liva method: {sequence_entropy(lambda_other)}")
    print(f"P(Y^n): {1/np.prod(cs)}")

    plt.figure()
    source_entropy = - hmm.pi[0] * np.log2(hmm.pi[0]) - hmm.pi[1] * np.log2(hmm.pi[1])
    plt.plot(np.arange(sim_length), entropies, label=r"$H(X_n \mid Y^n=y^n)$")
    plt.plot(np.arange(sim_length), np.ones(sim_length) * source_entropy, linestyle="-.", label="$H(X)$")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/sequence_entropy.png", bbox_inches="tight", pad_inches=0)
    plt.show()



