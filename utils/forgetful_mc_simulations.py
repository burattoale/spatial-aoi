import numpy as np
from collections import deque

from environment import DTMC, NodeDistribution, Node
from utils import SimulationParameters
from utils import lam, prob_x_given_y_0, h_y_delta
from utils import prob_x_given_y_0_spatial
from utils import PoiBin

def run_monte_carlo_simulation(params: SimulationParameters,
                               num_time_steps: int,
                               num_burn_in_steps: int,
                               seed: int = None,
                               zeta_bucket:bool = False):
    """
    Runs the Monte Carlo simulation and returns average entropy and its evolution.
    """
    rng = np.random.default_rng(seed)
    if seed is None:
        simulation_seed, dtmc_seed, node_dist_seed = rng.integers(0, 2**32-1, 3, dtype=np.uint32)
    else:
        simulation_seed, dtmc_seed, node_dist_seed = seed, seed, seed


    # 1. Initialize DTMC Source
    dtmc = DTMC(q=params.q, eta=params.eta, seed=int(dtmc_seed))

    # 2. Initialize Node Distribution
    node_dist = NodeDistribution(rho=params.rho,
                                 unit_radius=params.R_unit,
                                 K=params.K,
                                 zeta = params.zeta,
                                 alpha=params.alpha,
                                 beta=params.beta,
                                 seed=int(node_dist_seed),
                                 zeta_bucket=zeta_bucket)

    num_nodes = len(node_dist)
    if params.m_override is not None:
        num_nodes = params.m_override
    
    if num_nodes == 0:
        print("Warning: No nodes in the system (m=0).")
        # If m=0, conditional entropy will likely be H(X) if AoI keeps increasing.
        # For plotting, we need to decide what h_contrib should be.
        # Let's assume H(X) is the value if no info is ever received.
        pi_s = dtmc.pi
        h_source = -np.sum(pi_s * np.log2(pi_s, where=pi_s > 0))
        entropy_evolution = [h_source] * num_time_steps # Constant H(X)
        return h_source, entropy_evolution


    print(f"Simulation started with {num_nodes} nodes.")

    # 3. Initialize Receiver State
    last_received_y = 0# int(rng.choice(params.Y_symbols)) # Initial guess
    current_aoi = 0  # Age of Information, start fresh

    # Data collectors
    entropy_evolution = np.empty(num_time_steps, dtype=float) # To store H(X_t | Y_n, Delta_n) at each step after burn-in
    total_entropy_contribution = 0.0
    num_valid_steps_for_avg = 0

    # extract features from nodes
    nodes_lam = np.array([node.lam for node in node_dist.nodes])
    nodes_zeta = np.array([node.zeta for node in node_dist.nodes])
    prob_nodes_attempts_and_succeed_tx = nodes_zeta * (1 - params.epsilon)

    # cache for transition matrix elevated to the AoI power
    matrix_power_cache = {}
    matrix_power_cache[0] = np.eye(dtmc.A.shape[0])
    matrix_power_cache[1] = dtmc.A.copy()

    lambda_vector = [lam(d, params.alpha, params.R_unit) for d in range(params.K)]
    p_d_vector = [(2 * d + 1) / (params.K**2) for d in range(params.K)]
    lambda_vector = np.array(lambda_vector, dtype=float)
    p_d_vector = np.array(p_d_vector, dtype=float)

    # Simulation loop (Burn-in + Collection)
    for t in range(num_burn_in_steps + num_time_steps):
        # A. Source Evolution
        current_x_source_state = dtmc.step()

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
            successful_node = tx_idx[0]
            current_y = nodes_perception[successful_node]
            last_received_y = current_y
            current_aoi = 0
        else:
            current_aoi += 1

        # D. Entropy Contribution (after burn-in)
        if t >= num_burn_in_steps:
            # P(X_0=x | Y_0=last_received_y) vector for x in [0, 1]
            # This is P(X at time of Y reception | Y received)
            # Uses dtmc.pi (stationary distribution) for P(X=x) in Bayes rule
            p_x_given_y0_at_reception_vec = np.array([
                prob_x_given_y_0(x_val, last_received_y, dtmc.pi,
                                 params.K, params.R_unit, params.alpha, node_dist.tx_prob_bucket, lambda_vector, p_d_vector)
                for x_val in params.X_symbols
            ])

            if current_aoi not in matrix_power_cache:
                matrix_power_cache[current_aoi] = np.linalg.matrix_power(dtmc.A, current_aoi)
            A_aoi_pow = matrix_power_cache[current_aoi]

            # H(X_t | Y_n, Delta_n)
            # = H(X_{t_reception + Delta_n} | Y_n_at_t_reception = last_received_y)
            h_contrib = h_y_delta(p_x_given_y0_at_reception_vec,
                                  A_aoi_pow,
                                  x_symbols=np.array(params.X_symbols))
            
            entropy_evolution[t-num_burn_in_steps] = h_contrib
            total_entropy_contribution += h_contrib
            num_valid_steps_for_avg += 1

    estimated_avg_entropy = total_entropy_contribution / num_valid_steps_for_avg if num_valid_steps_for_avg > 0 else 0.0
    
    # If num_nodes was 0 and we returned early, entropy_evolution is already set.
    # If simulation ran but num_valid_steps_for_avg is 0 (e.g., num_time_steps=0), handle this.
    if num_valid_steps_for_avg == 0 and num_nodes > 0:
         # This might happen if num_time_steps is 0.
         # Calculate source entropy as a fallback for plotting.
        pi_s = dtmc.pi
        h_source_fallback = -np.sum(pi_s * np.log2(pi_s, where=pi_s > 0))
        entropy_evolution = [h_source_fallback] # or empty list, depends on desired plot behavior
        estimated_avg_entropy = h_source_fallback # Or NaN, or 0.0
        print("Warning: No valid steps for entropy averaging after burn-in.")


    return estimated_avg_entropy, entropy_evolution

# vertion with location aware mechanism
def run_monte_carlo_simulation_spatial(params: SimulationParameters,
                               num_time_steps: int,
                               num_burn_in_steps: int,
                               seed: int = None,
                               zeta_bucket:bool = False,
                               discard_logic:bool = False):
    """
    Runs the Monte Carlo simulation and returns average entropy and its evolution.
    """
    rng = np.random.default_rng(seed)
    if seed is None:
        simulation_seed, dtmc_seed, node_dist_seed = rng.integers(0, 2**32-1, 3, dtype=np.uint32)
    else:
        simulation_seed, dtmc_seed, node_dist_seed = seed, seed, seed


    # 1. Initialize DTMC Source
    dtmc = DTMC(q=params.q, eta=params.eta, seed=int(dtmc_seed))

    # 2. Initialize Node Distribution
    node_dist = NodeDistribution(rho=params.rho,
                                 unit_radius=params.R_unit,
                                 K=params.K,
                                 zeta = params.zeta,
                                 alpha=params.alpha,
                                 beta=params.beta,
                                 seed=int(node_dist_seed),
                                 zeta_bucket=zeta_bucket)

    num_nodes = len(node_dist)
    if params.m_override is not None:
        num_nodes = params.m_override
    
    if num_nodes == 0:
        print("Warning: No nodes in the system (m=0).")
        # If m=0, conditional entropy will likely be H(X) if AoI keeps increasing.
        # For plotting, we need to decide what h_contrib should be.
        # Let's assume H(X) is the value if no info is ever received.
        pi_s = dtmc.pi
        h_source = -np.sum(pi_s * np.log2(pi_s, where=pi_s > 0))
        entropy_evolution = [h_source] * num_time_steps # Constant H(X)
        return h_source, entropy_evolution


    print(f"Simulation started with {num_nodes} nodes.")

    # 3. Initialize Receiver State
    last_received_y = deque(maxlen=2)# int(rng.choice(params.Y_symbols)) # Initial guess
    last_received_y.appendleft(0)
    current_aoi = 0  # Age of Information, start fresh

    # Data collectors
    entropy_evolution = np.empty(num_time_steps, dtype=float) # To store H(X_t | Y_n, Delta_n) at each step after burn-in
    total_entropy_contribution = 0.0
    num_valid_steps_for_avg = 0

    # extract features from nodes
    nodes_lam = np.array([node.lam for node in node_dist.nodes])
    nodes_zeta = np.array([node.zeta for node in node_dist.nodes])
    prob_nodes_attempts_and_succeed_tx = nodes_zeta * (1 - params.epsilon)

    # cache for transition matrix elevated to the AoI power
    matrix_power_cache = {}
    matrix_power_cache[0] = np.eye(dtmc.A.shape[0])
    matrix_power_cache[1] = dtmc.A.copy()

    lambda_vector = [lam(d, params.alpha, params.R_unit) for d in range(params.K)]
    p_d_vector = [(2 * d + 1) / (params.K**2) for d in range(params.K)]
    lambda_vector = np.array(lambda_vector, dtype=float)
    p_d_vector = np.array(p_d_vector, dtype=float)
    prob_d_given_tx_vector = p_d_vector * node_dist.tx_prob_bucket
    prob_d_given_tx_vector /= np.sum(prob_d_given_tx_vector)

    # probability of success
    poibin_dist = PoiBin(np.array(node_dist.tx_probabilities))
    p_succ = poibin_dist.pmf(1)

    # Simulation loop (Burn-in + Collection)
    for t in range(num_burn_in_steps + num_time_steps):
        # A. Source Evolution
        current_x_source_state = dtmc.step()

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

        if t < num_burn_in_steps:
            if num_succ_tx == 1:
                successful_node_idx = tx_idx[0]
                successful_node:Node = node_dist[successful_node_idx]
                current_y = nodes_perception[successful_node_idx] * params.K + successful_node.zone_idx
                last_received_y.appendleft(current_y)
                current_aoi = 0
            else:
                current_aoi += 1

        # D. Entropy Contribution (after burn-in)
        if t >= num_burn_in_steps:

            if current_aoi not in matrix_power_cache:
                matrix_power_cache[current_aoi] = np.linalg.matrix_power(dtmc.A, current_aoi)
            if current_aoi+1 not in matrix_power_cache:
                matrix_power_cache[current_aoi+1] = np.linalg.matrix_power(dtmc.A, current_aoi+1)

            # Implement the selective drop logic
            # If the newly received symbol has lower entropy accept it,
            # otherwise discard it and continue without receiving it
            if num_succ_tx == 1 and discard_logic:
                successful_node_idx = tx_idx[0]
                successful_node:Node = node_dist[successful_node_idx]
                current_y = nodes_perception[successful_node_idx] * params.K + successful_node.zone_idx
                last_received_y.appendleft(current_y)
                p_x_given_y0_at_new_reception_vec = np.array([
                    prob_x_given_y_0_spatial(x_val, last_received_y[0], dtmc.pi,
                                     params.K, p_succ, prob_d_given_tx_vector, lambda_vector)
                    for x_val in params.X_symbols
                ])
                p_x_given_y0_at_old_reception_vec = np.array([
                    prob_x_given_y_0_spatial(x_val, last_received_y[1], dtmc.pi,
                                     params.K, p_succ, prob_d_given_tx_vector, lambda_vector)
                    for x_val in params.X_symbols
                ])

                incoming_entropy = h_y_delta(p_x_given_y0_at_new_reception_vec,
                                  np.linalg.matrix_power(dtmc.A, 0),
                                  x_symbols=np.array(params.X_symbols))
                old_entropy = h_y_delta(p_x_given_y0_at_old_reception_vec,
                                  matrix_power_cache[current_aoi+1],
                                  x_symbols=np.array(params.X_symbols))
                
                if incoming_entropy < old_entropy:
                    h_contrib = incoming_entropy
                    current_aoi = 0
                else:
                    h_contrib = old_entropy
                    current_aoi += 1
                    last_received_y.popleft()

            elif num_succ_tx == 1 and not discard_logic:
                successful_node_idx = tx_idx[0]
                successful_node:Node = node_dist[successful_node_idx]
                current_y = nodes_perception[successful_node_idx] * params.K + successful_node.zone_idx
                last_received_y.appendleft(current_y)
                p_x_given_y0_at_new_reception_vec = np.array([
                    prob_x_given_y_0_spatial(x_val, last_received_y[0], dtmc.pi,
                                     params.K, p_succ, prob_d_given_tx_vector, lambda_vector)
                    for x_val in params.X_symbols
                ])
                incoming_entropy = h_y_delta(p_x_given_y0_at_new_reception_vec,
                                  np.linalg.matrix_power(dtmc.A, 0),
                                  x_symbols=np.array(params.X_symbols))
                
                h_contrib = incoming_entropy
                current_aoi = 0

            else:
                current_aoi += 1
                p_x_given_y0_at_reception_vec = np.array([
                    prob_x_given_y_0_spatial(x_val, last_received_y[0], dtmc.pi,
                                     params.K, p_succ, prob_d_given_tx_vector, lambda_vector)
                    for x_val in params.X_symbols
                ])
                if current_aoi not in matrix_power_cache:
                    matrix_power_cache[current_aoi] = np.linalg.matrix_power(dtmc.A, current_aoi)
                A_aoi_pow = matrix_power_cache[current_aoi]
                h_contrib = h_y_delta(p_x_given_y0_at_reception_vec,
                                      A_aoi_pow,
                                      x_symbols=np.array(params.X_symbols))
            
            entropy_evolution[t-num_burn_in_steps] = h_contrib
            total_entropy_contribution += h_contrib
            num_valid_steps_for_avg += 1

    estimated_avg_entropy = total_entropy_contribution / num_valid_steps_for_avg if num_valid_steps_for_avg > 0 else 0.0
    
    # If num_nodes was 0 and we returned early, entropy_evolution is already set.
    # If simulation ran but num_valid_steps_for_avg is 0 (e.g., num_time_steps=0), handle this.
    if num_valid_steps_for_avg == 0 and num_nodes > 0:
         # This might happen if num_time_steps is 0.
         # Calculate source entropy as a fallback for plotting.
        pi_s = dtmc.pi
        h_source_fallback = -np.sum(pi_s * np.log2(pi_s, where=pi_s > 0))
        entropy_evolution = [h_source_fallback] # or empty list, depends on desired plot behavior
        estimated_avg_entropy = h_source_fallback # Or NaN, or 0.0
        print("Warning: No valid steps for entropy averaging after burn-in.")


    return estimated_avg_entropy, entropy_evolution