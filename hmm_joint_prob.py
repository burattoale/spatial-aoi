import numpy as np
from matplotlib import pyplot as plt
import math
import copy
from tqdm import tqdm

from environment import HiddenMM, SpatialHMM
from environment import NodeDistribution, Node
from utils import SimulationParameters
from utils import cond_prob_y_given_x, sequence_entropy

def hmm_entropy(params:SimulationParameters, simulation_length:int=10000, loc_aware=False):
    local_params:SimulationParameters = copy.deepcopy(params)
    if not loc_aware:
        local_params.Y_symbols = [0, 1, 2, 3]
        hmm = HiddenMM(local_params)
    else:
        hmm = SpatialHMM(local_params)
    Y = np.empty(simulation_length, dtype=object) # observations
    X_true = np.empty_like(Y) # real state of the system
    alpha_vec = np.empty(2) # forward variables
    cs = np.zeros(simulation_length) # scaling coefficients
    entropies = np.zeros(simulation_length)
    pred_X = np.empty_like(X_true)

    print(f"Starting simulation with {simulation_length} steps")
    print(f"Number of sensors: {local_params.m}")
    # initialize the first values of the simulation
    X_true[0] = hmm.hidden_state
    Y[0] = hmm.hidden_state
    cond_prob0 = hmm.B[0, Y[0]]
    cond_prob1 = hmm.B[1, Y[0]]
    alpha_vec[0] = hmm.pi[0] * cond_prob0
    alpha_vec[1] = hmm.pi[1] * cond_prob1
    cs[0] = 1
    cumulative_cs_prod = cs[0]
    pred_X[0] = np.argmax(alpha_vec/cumulative_cs_prod)

    # Viterbi algorithm initialization
    X_path = np.empty_like(Y, dtype=int) # state sequence
    phi_matrix = np.zeros((simulation_length, len(local_params.X_symbols)))
    psi_matrix = np.empty_like(phi_matrix)
    for x in local_params.X_symbols:
        phi_matrix[0][x] = np.log(hmm.pi[x]) + np.log(hmm.B[x, Y[0]]+1e-12)
        psi_matrix[0][x] = 0

    for t in tqdm(range(1, simulation_length)):
        Y[t] = hmm.step()
        X_true[t] = hmm.hidden_state

        temp0, temp1 = 0, 0
        for id in range(2): # only two states
            temp0 += alpha_vec[id] * hmm.A[id, 0]
            temp1 += alpha_vec[id] * hmm.A[id, 1]
        alpha_vec[0] = temp0 * hmm.B[0, Y[t]]
        alpha_vec[1] = temp1 * hmm.B[1, Y[t]]
        cs[t] = np.sum(alpha_vec)
        alpha_vec /= cs[t] # normalize the forward probabilities

        p_0_y_norm = alpha_vec[0] / np.sum(alpha_vec)
        p_1_y_norm = alpha_vec[1] / np.sum(alpha_vec)
        entropies[t] = - p_0_y_norm * np.log2(p_0_y_norm+1e-12) - p_1_y_norm * np.log2(p_1_y_norm+1e-12)
        pred_X[t] = np.argmax(np.array(p_0_y_norm, p_1_y_norm))

        # viterbi algorithm recursion step
        for j in local_params.X_symbols:
            temp_max_argument = np.empty(len(local_params.X_symbols))
            for i in local_params.X_symbols:
                temp_max_argument[i] = phi_matrix[t-1][i] + np.log(hmm.A[i, j]+1e-12)
            phi_matrix[t][j] = np.max(temp_max_argument) + np.log(hmm.B[j, Y[t]]+1e-12)
            psi_matrix[t][j] = np.argmax(temp_max_argument)

    # Viterbi algorithm path reconstruction
    X_path[-1] = np.argmax(phi_matrix[-1])
    for t in reversed(range(simulation_length-1)):
        X_path[t] = psi_matrix[t+1][X_path[t+1]]
    
    estimation_error_prob = np.sum(np.abs(X_path - X_true)) / simulation_length
    short_est_prob = np.sum(np.abs(X_path - X_true)) / simulation_length

    return entropies, np.mean(entropies), estimation_error_prob, short_est_prob


def run_hmm_simulation(params: SimulationParameters,
                               num_time_steps: int,
                               loc_aware:bool=False,
                               non_binary:bool=False,
                               fixed_nodes_per_region:bool=False,
                               seed: int = None):
    """
    Runs the Monte Carlo simulation and returns average entropy and its evolution.
    """
    rng = np.random.default_rng(seed)
    if seed is None:
        simulation_seed, hmm_seed, node_dist_seed = rng.integers(0, 2**32-1, 3, dtype=np.uint32)
    else:
        simulation_seed, hmm_seed, node_dist_seed = seed, seed, seed


    # 1. Initialize HMM Source
    local_params:SimulationParameters = copy.deepcopy(params)
    local_params.beta = 0
    if not loc_aware and not non_binary:
        local_params.Y_symbols = [0, 1, 2, 3]
        hmm = HiddenMM(local_params, hmm_seed)
    elif non_binary and not loc_aware:
        hmm = HiddenMM(local_params, hmm_seed)
    else:
        hmm = SpatialHMM(local_params, hmm_seed)

    # 2. Initialize Node Distribution
    node_dist = NodeDistribution(rho=local_params.rho,
                                 unit_radius=local_params.R_unit,
                                 K=local_params.K,
                                 zeta = local_params.zeta,
                                 alpha=local_params.alpha,
                                 beta=local_params.beta,
                                 seed=int(node_dist_seed),
                                 zeta_bucket=True,
                                 fixed_nodes_per_region=fixed_nodes_per_region)

    num_nodes = len(node_dist)
    if local_params.m_override is not None:
        num_nodes = local_params.m_override
    
    if num_nodes == 0:
        print("Warning: No nodes in the system (m=0).")
        # If m=0, conditional entropy will likely be H(X) if AoI keeps increasing.
        # For plotting, we need to decide what h_contrib should be.
        # Let's assume H(X) is the value if no info is ever received.
        pi_s = hmm.pi
        h_source = -np.sum(pi_s * np.log2(pi_s, where=pi_s > 0))
        entropy_evolution = [h_source] * num_time_steps # Constant H(X)
        return h_source, entropy_evolution


    # print(f"Simulation started with {num_nodes} nodes.")
    # if fixed_nodes_per_region:
    #     print(f"Nodes per region: {node_dist.nodes_per_region}")
    #     print(f"Total number of fixed nodes: {np.sum(node_dist.nodes_per_region)}")

    # 3. Initialize Receiver State
    X_true = np.empty(num_time_steps, dtype=int)
    Y = np.empty(num_time_steps, dtype=int)
    cs = np.empty_like(Y, dtype=float)
    alpha_vec = np.zeros(len(local_params.X_symbols))
    last_received_y = 0# int(rng.choice(local_params.Y_symbols)) # Initial guess
    pred_X = np.empty_like(Y)

    # Data collectors
    entropy_evolution = np.empty(num_time_steps, dtype=float) # To store H(X_t | Y_n, Delta_n) at each step after burn-in
    total_entropy_contribution = 0.0
    num_valid_steps_for_avg = 0

    # extract features from nodes
    nodes_lam = np.array([node.lam for node in node_dist.nodes])
    nodes_zeta = np.array([node.zeta for node in node_dist.nodes])
    prob_nodes_attempts_and_succeed_tx = nodes_zeta * (1 - local_params.epsilon)

    X_true[0] = hmm.hidden_state
    Y[0] = int(rng.choice(local_params.Y_symbols, p=hmm.B[X_true[0]])) # initialize first received symbol
    cond_prob_vec = hmm.B[:, Y[0]]
    alpha_vec = hmm.pi * cond_prob_vec
    cs[0] = 1
    pred_X[0] = np.argmax(alpha_vec)

    # Viterbi algorithm initialization
    X_path = np.empty_like(Y, dtype=int) # state sequence
    phi_matrix = np.zeros((num_time_steps, len(local_params.X_symbols)))
    psi_matrix = np.empty_like(phi_matrix)
    for x in local_params.X_symbols:
        phi_matrix[0][x] = np.log(hmm.pi[x]) + np.log(hmm.B[x, Y[0]]+1e-12)
        psi_matrix[0][x] = 0

    # Simulation loop
    for t in range(1, num_time_steps):
        # A. Source Evolution
        hmm.step()
        X_true[t] = hmm.hidden_state
        current_x_source_state = hmm.hidden_state

        # B. Node Observation and Transmission
        rand_perception = rng.random(num_nodes)
        rand_tx = rng.random(num_nodes)
        perc_correct = current_x_source_state
        perc_not_correct = 1 - current_x_source_state

        nodes_perception = np.where(
            rand_perception < nodes_lam,
            perc_correct,
            perc_not_correct
        )

        tx_mask = rand_tx < prob_nodes_attempts_and_succeed_tx
        tx_idx = np.nonzero(tx_mask)[0]
        num_succ_tx = len(tx_idx)

        if num_succ_tx == 1:
            successful_node_id = tx_idx[0]
            successul_node:Node = node_dist[successful_node_id]
            # the symbol is the node zone if symbol is 0 
            # and K + node's zone if symbol is 1
            if loc_aware:
                current_y = nodes_perception[successful_node_id] * local_params.K + successul_node.zone_idx
            else:
                current_y = nodes_perception[successful_node_id]
            last_received_y = current_y
        else:
            if loc_aware:
                last_received_y = 2 * local_params.K + 1
            else:
                last_received_y = local_params.Y_symbols[-1]

        # D. Entropy Contribution
        Y[t] = last_received_y
        # compute entropy
        temp_vec = np.zeros(len(local_params.X_symbols))
        for id in range(len(local_params.X_symbols)): # only two states
            temp_vec[id] = np.sum(alpha_vec * hmm.A[:,id])
        alpha_vec = temp_vec * hmm.B[:, Y[t]]
        cs[t] = np.sum(alpha_vec)
        alpha_vec /= cs[t] # normalize the forward probabilities

        p_x_y_norm = alpha_vec / np.sum(alpha_vec)
        entropy = - np.sum(p_x_y_norm * np.log2(p_x_y_norm+1e-12))
        pred_X = np.argmax(p_x_y_norm)
            
        entropy_evolution[t] = entropy
        total_entropy_contribution += entropy
        num_valid_steps_for_avg += 1

        # viterbi algorithm recursion step
        for j in local_params.X_symbols:
            temp_max_argument = np.empty(len(local_params.X_symbols))
            for i in local_params.X_symbols:
                temp_max_argument[i] = phi_matrix[t-1][i] + np.log(hmm.A[i, j]+1e-12)
            phi_matrix[t][j] = np.max(temp_max_argument) + np.log(hmm.B[j, Y[t]]+1e-12)
            psi_matrix[t][j] = np.argmax(temp_max_argument)
    
    # Viterbi algorithm path reconstruction
    X_path[-1] = np.argmax(phi_matrix[-1])
    for t in reversed(range(num_time_steps-1)):
        X_path[t] = psi_matrix[t+1][X_path[t+1]]
    
    estimation_error_prob = np.sum(np.abs(X_path - X_true)) / num_time_steps
    short_est_prob = np.sum(np.abs(X_path - X_true)) / num_time_steps

    estimated_avg_entropy = total_entropy_contribution / num_valid_steps_for_avg if num_valid_steps_for_avg > 0 else 0.0
    
    # If num_nodes was 0 and we returned early, entropy_evolution is already set.
    # If simulation ran but num_valid_steps_for_avg is 0 (e.g., num_time_steps=0), handle this.
    if num_valid_steps_for_avg == 0 and num_nodes > 0:
         # This might happen if num_time_steps is 0.
         # Calculate source entropy as a fallback for plotting.
        pi_s = hmm.pi
        h_source_fallback = -np.sum(pi_s * np.log2(pi_s, where=pi_s > 0))
        entropy_evolution = [h_source_fallback] # or empty list, depends on desired plot behavior
        estimated_avg_entropy = h_source_fallback # Or NaN, or 0.0
        print("Warning: No valid steps for entropy averaging after burn-in.")

    return entropy_evolution, estimated_avg_entropy, estimation_error_prob, short_est_prob, Y


if __name__ == "__main__":
    sim_length = 20000
    # sim parameters
    q = 0.005
    eta = 1
    zeta = 0.01
    epsilon = 0.1
    rho = 5e-2
    R = 10
    K = 2
    m = math.floor(rho * np.pi * (R*K)**2)
    print(f"Number of sensors: {m}")
    alpha = 0.25
    x_symbols = [0, 1]
    y_symbols = [0, 1, 2, 3]
    # spatial HMM
    y_symbols = [i for i in range(len(x_symbols) * K + 2)]
    y_translation = {
        0:"0",
        1:"1",
        2:"C",
        3:"I"
    }

    local_params = SimulationParameters(q, eta, zeta, epsilon, rho=rho, m=m, K=K, alpha=alpha, R_unit=R, X_symbols=x_symbols, Y_symbols=y_symbols)

    entropies, _, est_error_no_loc, _ = hmm_entropy(local_params, sim_length, loc_aware=False)
    mc_entropies, _, _, _, _ = run_hmm_simulation(local_params, sim_length, loc_aware=False)



    #lambda_n = np.log(alpha_vec[0]/alpha_vec[1])
    # print(f"Entropy alpha method: {sequence_entropy(lambda_n)}")
    #print(f"Entropy Liva method: {sequence_entropy(lambda_other)}")


    plt.figure()
    #source_entropy = - hmm.pi[0] * np.log2(hmm.pi[0]) - hmm.pi[1] * np.log2(hmm.pi[1])
    plt.plot(np.arange(sim_length), entropies, label=r"$H(X_n \mid Y^n=y^n)$")
    plt.plot(np.arange(sim_length), mc_entropies, label=r"$H(X_n \mid Y^n=y^n)$ Monte Carlo")
    #plt.plot(np.arange(sim_length), np.ones(sim_length) * source_entropy, linestyle="-.", label="$H(X)$")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/sequence_entropy_1.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    entropies, _, est_error_yes_loc, _ = hmm_entropy(local_params, sim_length, loc_aware=True)
    mc_entropies, _, _, _, _ = run_hmm_simulation(local_params, sim_length, loc_aware=True)

    print(f"Estimation error NO localization aware: {est_error_no_loc}")
    print(f"Estimation error YES localization aware: {est_error_yes_loc}")
    #lambda_n = np.log(alpha_vec[0]/alpha_vec[1])
    # print(f"Entropy alpha method: {sequence_entropy(lambda_n)}")
    #print(f"Entropy Liva method: {sequence_entropy(lambda_other)}")


    plt.figure()
    #source_entropy = - hmm.pi[0] * np.log2(hmm.pi[0]) - hmm.pi[1] * np.log2(hmm.pi[1])
    plt.plot(np.arange(sim_length), entropies, label=r"$H(X_n \mid Y^n=y^n)$")
    plt.plot(np.arange(sim_length), mc_entropies, label=r"$H(X_n \mid Y^n=y^n)$, Monte Carlo")
    #plt.plot(np.arange(sim_length), np.ones(sim_length) * source_entropy, linestyle="-.", label="$H(X)$")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/sequence_entropy_2.png", bbox_inches="tight", pad_inches=0)
    #plt.show()
    plt.close()

    plt.figure(0)
    entropy_no_loc, _, est_error_no_loc, est_short_no_loc, Y1 = run_hmm_simulation(local_params, 10000, loc_aware=False, seed=0)
    entropy_yes_loc, _, est_error_yes_loc, est_short_yes_loc, Y2 = run_hmm_simulation(local_params, 10000, loc_aware=True, seed=0)
    plt.plot(entropy_no_loc, label="Entropy no location")
    plt.plot(entropy_yes_loc, label="Entropy location aware")
    plt.savefig("plots/compare_yes_no_location.png", bbox_inches="tight", pad_inches=0)
    print(Y1)
    print(Y2)
    print(f"Estimation error NO localization aware: {est_error_no_loc}, {est_short_no_loc}")
    print(f"Estimation error YES localization aware: {est_error_yes_loc}, {est_short_yes_loc}")
    plt.legend()
    plt.grid()
    #plt.show()
    plt.close()

    Ks = np.arange(2,3)
    sim_length = 10000
    errors_no_loc = []
    errors_short_no_loc = []
    errors_yes_loc = []
    errors_short_yes_loc = []
    for k in Ks:
        local_params.K = int(k)
        local_params.Y_symbols = [i for i in range(len(x_symbols) * k + 2)]
        _, _, e_no_loc, e_short_no_loc, _ = run_hmm_simulation(local_params, sim_length, loc_aware=False, seed=0)
        _, _, e_yes_loc, e_short_yes_loc, _ = run_hmm_simulation(local_params, sim_length, loc_aware=True, seed=0)
        errors_no_loc.append(e_no_loc)
        errors_yes_loc.append(e_yes_loc)
        errors_short_no_loc.append(e_short_no_loc)
        errors_short_yes_loc.append(e_short_yes_loc)

    plt.figure()
    plt.semilogy(Ks, errors_no_loc, marker='.', label="HMM Viterbi")
    plt.semilogy(Ks, errors_yes_loc, marker='.', label="HMM loc. aware Viterbi")
    plt.semilogy(Ks, errors_short_no_loc, marker="x", label="HMM limited seq.")
    plt.semilogy(Ks, errors_short_yes_loc, marker="x", label="HMM loc. aware limited seq.")
    plt.xlabel("Number of sections, K")
    plt.ylabel("Estimation error probability")
    plt.legend()
    plt.grid()
    plt.show()
