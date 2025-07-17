from utils import prob_x_given_y_0, SimulationParameters, lam
from environment import DTMC

import numpy as np

params = SimulationParameters(
        q=0.005,
        eta=10,
        zeta=5e-4, # Example: list of base probabilities. NodeDistribution will use this.
        epsilon=0.1,
        rho=0.05,
        m_override=None,
        K=2, # Initial K for the first plot
        alpha=0.02,
        beta=0, # Default beta, will be overridden in the loop
        R_unit=10
    )
dtmc = DTMC(params.q, params.eta, seed=0)
lambda_vector = [lam(d, params.alpha, params.R_unit) for d in range(params.K)]
p_d_vector = [(2 * d + 1) / (params.K**2) for d in range(params.K)]
lambda_vector = np.array(lambda_vector, dtype=float)
p_d_vector = np.array(p_d_vector, dtype=float)
p_x_vector_0 = np.array([prob_x_given_y_0(x, 1, dtmc.pi, params.K, params.R_unit, params.alpha, np.array([params.zeta]*params.K), lambda_vector, p_d_vector) for x in [0,1]])
entropy = 0
for t in range(50):
    p_x_vector = p_x_vector_0 @ np.linalg.matrix_power(dtmc.A, t)
    entropy = - np.sum(p_x_vector * np.log2(p_x_vector))
    print(entropy)