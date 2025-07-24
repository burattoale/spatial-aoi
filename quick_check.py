from utils import prob_x_given_y_0, SimulationParameters, lam, generate_lambda_matrix
from environment import DTMC
from hmm_joint_prob import run_hmm_simulation

import numpy as np
from copy import deepcopy
import pickle
from tqdm import tqdm

params = SimulationParameters(
        q=0.005,
        eta=10,
        zeta=5e-4, # Example: list of base probabilities. NodeDistribution will use this.
        epsilon=0.1,
        rho=0.05,
        m_override=None,
        K=12, # Initial K for the first plot
        alpha=0.02,
        beta=0, # Default beta, will be overridden in the loop
        R_unit=10
    )
dtmc = DTMC(params.q, params.eta, seed=0)
lambda_mat = generate_lambda_matrix(len(params.X_symbols), params.K, params.alpha, params.R_unit)
p_d_vector = [(2 * d + 1) / (params.K**2) for d in range(params.K)]
p_d_vector = np.array(p_d_vector, dtype=float)
p_x_vector_0 = np.array([prob_x_given_y_0(x, 1, dtmc.pi, params.K, np.array([params.zeta]*params.K), lambda_mat, p_d_vector) for x in params.X_symbols])
entropy = 0
for t in range(50):
    p_x_vector = p_x_vector_0 @ np.linalg.matrix_power(dtmc.A, t)
    entropy = - np.sum(p_x_vector * np.log2(p_x_vector))
    print(entropy)

# evaluate two extreme configurations
params.Y_symbols = [0,1,2]
params1 = deepcopy(params)
params2 = deepcopy(params)
with open("results/zeta_optim_hmm_12.pickle", "rb") as f:
    data:dict = pickle.load(f)
    p_vals = list(data[list(data.keys())[11]]["zeta"].values())
params1.zeta = np.array(p_vals)
params2.zeta = np.array([5e-4] * params.K)
print(params1.zeta)

avg1 = []
avg2 = []
for i in tqdm(range(80)):
    _, avg_entropy1, _, _, _ = run_hmm_simulation(params1, 10000, fixed_nodes_per_region=True, seed=i)
    _, avg_entropy2, _, _, _ = run_hmm_simulation(params2, 10000, fixed_nodes_per_region=True, seed=i)
    avg1.append(avg_entropy1)
    avg2.append(avg_entropy2)

print(f"Optim: {np.mean(avg1)}")
print(f"Full 5e-4: {np.mean(avg2)}")