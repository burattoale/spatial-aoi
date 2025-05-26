# main_simulation.py
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from collections import defaultdict
from tqdm import tqdm

from utils import *
from environment import *



def run_monte_carlo_simulation(params: SimulationParameters,
                               num_time_steps: int,
                               num_burn_in_steps: int,
                               seed: int = None):
    """
    Runs the Monte Carlo simulation and returns average entropy and its evolution.
    """
    rng = np.random.default_rng(seed)
    simulation_seed, dtmc_seed, node_dist_seed = rng.integers(0, 2**32-1, 3, dtype=np.uint32)


    # 1. Initialize DTMC Source
    dtmc = DTMC(q=params.q, eta=params.eta, seed=int(dtmc_seed))

    # 2. Initialize Node Distribution
    node_dist = NodeDistribution(rho=params.rho,
                                 unit_radius=params.R_unit,
                                 K=params.K,
                                 alpha=params.alpha,
                                 beta=params.beta,
                                 seed=int(node_dist_seed))

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
    last_received_y = int(rng.choice(params.Y_symbols)) # Initial guess
    current_aoi = 0  # Age of Information, start fresh

    # Data collectors
    entropy_evolution = [] # To store H(X_t | Y_n, Delta_n) at each step after burn-in
    total_entropy_contribution = 0.0
    num_valid_steps_for_avg = 0

    # Simulation loop (Burn-in + Collection)
    for t in range(num_burn_in_steps + num_time_steps):
        # A. Source Evolution
        current_x_source_state = dtmc.step()

        # B. Node Observation and Transmission
        successful_observations = []

        for node_idx in range(num_nodes):
            # Ensure we don't go out of bounds if m_override > actual len(node_dist.nodes)
            actual_node_obj = node_dist.nodes[node_idx % len(node_dist.nodes)] 

            prob_node_attempts_and_succeeds_tx = actual_node_obj.zeta * (1 - params.epsilon)

            # Node perceives the source state (possibly incorrectly)
            if rng.random() < actual_node_obj.lam: # lam is P(correct perception)
                node_perceived_x = current_x_source_state
            else:
                node_perceived_x = 1 - current_x_source_state # Binary assumption

            # Node attempts to transmit; transmission can be erased
            if rng.random() < prob_node_attempts_and_succeeds_tx:
                successful_observations.append(node_perceived_x)

        # C. Receiver Update
        # Condition: if EXACTLY ONE node successfully transmitted (as per user's modified code)
        if len(successful_observations) == 1:
            current_y = successful_observations[0] # The only observation is the received one
            last_received_y = current_y
            current_aoi = 0
        else: # No successful transmission OR multiple successful transmissions (treated as no update based on "==1" condition)
            current_aoi += 1

        # D. Entropy Contribution (after burn-in)
        if t >= num_burn_in_steps:
            # P(X_0=x | Y_0=last_received_y) vector for x in [0, 1]
            # This is P(X at time of Y reception | Y received)
            # Uses dtmc.pi (stationary distribution) for P(X=x) in Bayes rule
            p_x_given_y0_at_reception_vec = np.array([
                prob_x_given_y_0(x_val, last_received_y, dtmc.pi,
                                 params.K, params.R_unit, params.alpha)
                for x_val in params.X_symbols
            ])

            # H(X_t | Y_n, Delta_n)
            # = H(X_{t_reception + Delta_n} | Y_n_at_t_reception = last_received_y)
            h_contrib = h_y_delta(p_x_given_y0_at_reception_vec,
                                  dtmc.A,
                                  current_aoi,
                                  x_symbols=np.array(params.X_symbols))
            
            entropy_evolution.append(h_contrib)
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

# Worker function for a single MC simulation task
def worker_mc_simulation_task(k_val_task, base_params_dict_task, sim_runs_task, burn_in_task, mc_seed_task):
    # Create SimulationParameters object for this task's configuration
    current_sim_config_params = SimulationParameters(
        q=base_params_dict_task['q'], eta=base_params_dict_task['eta'],
        zeta=base_params_dict_task['zeta'], # Base zeta, if NodeDistribution uses it
        epsilon=base_params_dict_task['epsilon'],
        rho=base_params_dict_task['rho'],
        m_override=base_params_dict_task['m_override'], # Pass m_override policy
        K=k_val_task, # Current K value for this task
        alpha=base_params_dict_task['alpha'],
        R_unit=base_params_dict_task['R_unit'],
        beta=base_params_dict_task['beta'],
        X_symbols=base_params_dict_task['X_symbols']
    )

    avg_entropy_mc, _ = run_monte_carlo_simulation(
        params=current_sim_config_params, # General params
        num_time_steps=sim_runs_task,
        num_burn_in_steps=burn_in_task,
        seed=mc_seed_task,
    )
    return k_val_task, avg_entropy_mc



if __name__ == "__main__":
    # Define Simulation Parameters
    params = SimulationParameters(
        q=0.005,          # DTMC: P(0->1)
        eta=1,        # DTMC: P(1->0) = eta*q
        zeta=[1e-4],      # Node: probability of attempting transmission
        epsilon=0.1,    # Channel: erasure probability
        rho=0.005,        # Node density for calculating m
        m_override=None,# Example: 10 # Set to an int to fix number of nodes, else calculated by rho
        K=5,            # Number of circular regions/buckets (initial value)
        alpha=0.02,     # Power law exponent for P(correct)
        beta=2,
        R_unit=10       # Radius of each unit section
    )

    # Monte Carlo settings
    num_simulation_runs = 50000
    num_burn_in = 1000
    simulation_seed = 0

    print(f"Starting Monte Carlo simulation with parameters: {params}")

    avg_entropy_mc, entropy_time_series = run_monte_carlo_simulation(
        params,
        num_time_steps=num_simulation_runs,
        num_burn_in_steps=num_burn_in,
        seed=simulation_seed
    )

    print(f"\nMonte Carlo Estimated Average H(X_t | Y_n, Delta_n): {avg_entropy_mc:.4f}")

    # --- Theoretical Calculations & Source Entropy ---
    dtmc_theoretical = DTMC(q=params.q, eta=params.eta, seed=simulation_seed)
    pi_source = dtmc_theoretical.pi
    h_x_source = -np.sum(pi_source * np.log2(pi_source, where=pi_source > 0))
    print(f"Entropy of source H(X): {h_x_source:.4f}")

    # Calculate m based on initial params.K
    _node_dist_for_m = NodeDistribution(rho=params.rho, unit_radius=params.R_unit, K=params.K, alpha=params.alpha, beta=params.beta)
    if params.m_override is None:
        m_calc = len(_node_dist_for_m)
    else:
        m_calc = params.m_override
    
    if m_calc > 0 :
        print(f"\nCalculating theoretical average entropy using m={m_calc} (for initial K={params.K}, can be slow)...")
        # Calculate P(success) for max_delta calculation
        zetas = _node_dist_for_m.tx_probabilities
        print(zetas)
        p_success_initial_k = ps_calc(m_calc, zetas, params.epsilon)
        max_delta_initial_k = max(50, int(2 / (p_success_initial_k + 1e-9)))

        theoretical_entropy = overall_entropy(
            A=dtmc_theoretical.A,
            pi=dtmc_theoretical.pi,
            K=params.K,
            R_unit=params.R_unit,
            alpha=params.alpha,
            m=m_calc,
            zeta=zetas,
            epsilon=params.epsilon,
            states=params.X_symbols, 
            max_delta_considered=max_delta_initial_k
        )
        print(f"Theoretical H(X_t | Y_n, Delta_n) (for K={params.K}): {theoretical_entropy:.4f}")
    else:
        print(f"\nSkipping theoretical entropy calculation for K={params.K} as m_calc = {m_calc}.")
        theoretical_entropy = h_x_source

    # --- Plotting (Original: Evolution over Time) ---
    if entropy_time_series:
        time_axis = np.arange(len(entropy_time_series))
        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, entropy_time_series, label=r'Instantaneous $H(X_t | Y_n, \Delta_n)$', alpha=0.7)
        plt.axhline(y=h_x_source, color='r', linestyle='--', label='$H(X)$ (Source Entropy)')
        plt.axhline(y=avg_entropy_mc, color='g', linestyle=':', label=r'Avg. Est. $H(X_t | Y_n, \Delta_n)$ (MC)')
        if m_calc > 0:
             plt.axhline(y=theoretical_entropy, color='purple', linestyle='-.', label=fr'Theoretical Avg. $H(X_t | Y_n, \Delta_n)$ (K={params.K})')
        
        plt.xlabel(f"Time step (after {num_burn_in} burn-in steps)")
        plt.ylabel("Entropy (bits)")
        plt.title(f"Evolution of Conditional Entropy over Time (K={params.K})")
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=0)
        if entropy_time_series:
            max_plot_y = max(h_x_source * 1.1, np.max(entropy_time_series) * 1.1 if entropy_time_series else h_x_source * 1.1)
            if m_calc > 0 : max_plot_y = max(max_plot_y, theoretical_entropy * 1.1)
            plt.ylim(top=max_plot_y)
        plt.tight_layout()
        plt.show()
    else:
        print("No entropy data to plot (entropy_time_series is empty).")

    # --- NEW PLOT: Entropy vs. K (Parallelized) ---
    print("\nGenerating plot for Entropy vs. K (with parallel simulations)...")

    K_start_plot = 4
    K_end_plot = 34
    K_values_for_plot = list(range(K_start_plot, K_end_plot + 1))
    
    simulation_K_interval = 1 
    num_mc_repetitions = 5    
    n_parallel_jobs = -1      

    num_simulation_runs_sweep = 10000 
    num_burn_in_sweep = 500      

    # --- 1. Theoretical Calculations (sequential for each K) ---
    theoretical_entropies_vs_K = []
    print("\n--- Calculating Theoretical Entropies (Sequentially) ---")
    for k_val_theoretical in tqdm(K_values_for_plot):
        
        current_params_theoretical = SimulationParameters(
            q=params.q, eta=params.eta, zeta=params.zeta, epsilon=params.epsilon,
            rho=params.rho, 
            m_override=params.m_override, # Pass m_override policy
            K=k_val_theoretical, 
            alpha=params.alpha, R_unit=params.R_unit,
            beta=params.beta, X_symbols=params.X_symbols 
        )

        # NodeDistribution respects m_override from current_params_theoretical
        # It's assumed NodeDistribution's __init__ takes m_override, or you pass it explicitly:
        _node_dist_k_theo = NodeDistribution(
            rho=current_params_theoretical.rho, 
            unit_radius=current_params_theoretical.R_unit, 
            K=current_params_theoretical.K, 
            alpha=current_params_theoretical.alpha,
            beta=current_params_theoretical.beta,
        )
        
        m_calc_k_theo = len(_node_dist_k_theo) # This m is consistent with m_override policy
        
        if m_calc_k_theo > 0:
            current_ps_success_theo = ps_calc(m_calc_k_theo, _node_dist_k_theo.tx_probabilities, current_params_theoretical.epsilon)
            print(f"PS theorical: {current_ps_success_theo}")
            max_delta_k_theo = 1000#max(100, int(2 / (current_ps_success_theo + 1e-9)))
            if current_ps_success_theo < 1e-7:
                 print(f"Warning (Theoretical): Low P(success) ({current_ps_success_theo:.2e}) for K={k_val_theoretical}, m={m_calc_k_theo}.")

            theoretical_H_k = overall_entropy(
                A=dtmc_theoretical.A, pi=dtmc_theoretical.pi,
                K=current_params_theoretical.K, R_unit=current_params_theoretical.R_unit,
                alpha=current_params_theoretical.alpha, m=m_calc_k_theo,
                zeta=_node_dist_k_theo.tx_probabilities, # Pass the list of probabilities
                epsilon=current_params_theoretical.epsilon, states=current_params_theoretical.X_symbols,
                max_delta_considered=max_delta_k_theo
            )
            theoretical_entropies_vs_K.append(theoretical_H_k)
        else:
            theoretical_entropies_vs_K.append(h_x_source)
    print("Theoretical calculations finished.")

    # --- 2. Determine K values requiring simulation runs ---
    k_values_to_simulate_explicitly = []
    # Simplified logic for simulation interval
    if simulation_K_interval > 0:
        k_values_to_simulate_explicitly = [k for k in K_values_for_plot if (k - K_start_plot) % simulation_K_interval == 0]
        if K_start_plot not in k_values_to_simulate_explicitly: # Ensure start is included
             k_values_to_simulate_explicitly.insert(0, K_start_plot)
        if K_end_plot not in k_values_to_simulate_explicitly and K_end_plot != k_values_to_simulate_explicitly[-1]: # Ensure end is included
             k_values_to_simulate_explicitly.append(K_end_plot)
        k_values_to_simulate_explicitly = sorted(list(set(k_values_to_simulate_explicitly)))
    elif K_values_for_plot : # Simulate only start and end if interval is 0 or less (and list is not empty)
        k_values_to_simulate_explicitly.append(K_start_plot)
        if K_end_plot != K_start_plot: k_values_to_simulate_explicitly.append(K_end_plot)
        k_values_to_simulate_explicitly = sorted(list(set(k_values_to_simulate_explicitly)))


    # --- 3. Parallel Monte Carlo Simulations ---
    simulated_K_points_aggregated = []
    simulated_entropies_aggregated = []

    if k_values_to_simulate_explicitly:
        print(f"\n--- Preparing for Parallel Monte Carlo Simulations ---")
        print(f"K values to simulate: {k_values_to_simulate_explicitly}")
        print(f"Number of repetitions per K: {num_mc_repetitions}")

        base_params_for_worker = {
            'q': params.q, 'eta': params.eta, 'zeta': params.zeta, 
            'epsilon': params.epsilon, 'rho': params.rho, 
            'm_override': params.m_override, 'alpha': params.alpha, 
            'R_unit': params.R_unit, 'beta': params.beta, 
            'X_symbols': params.X_symbols
        }

        simulation_job_args_list = []
        for k_to_sim in k_values_to_simulate_explicitly:
            for rep_idx in range(num_mc_repetitions):
                unique_run_seed = simulation_seed + (k_to_sim * 1000) + rep_idx 
                simulation_job_args_list.append(
                    (k_to_sim, base_params_for_worker, num_simulation_runs_sweep, num_burn_in_sweep, unique_run_seed)
                )
        
        print(f"Submitting {len(simulation_job_args_list)} MC simulation jobs for parallel execution ({n_parallel_jobs} cores)...")
        
        parallel_results_list = Parallel(n_jobs=n_parallel_jobs, verbose=5)(
            delayed(worker_mc_simulation_task)(*args) for args in simulation_job_args_list
        )

        mc_results_by_k_map = defaultdict(list)
        for k_result, entropy_result in parallel_results_list:
            mc_results_by_k_map[k_result].append(entropy_result)

        sorted_sim_k_values = sorted(mc_results_by_k_map.keys())
        for k_agg in sorted_sim_k_values:
            entropies_for_this_k = mc_results_by_k_map[k_agg]
            avg_entropy_for_this_k = np.mean(entropies_for_this_k)
            simulated_K_points_aggregated.append(k_agg)
            simulated_entropies_aggregated.append(avg_entropy_for_this_k)
            print(f"Aggregated MC Result for K={k_agg}: Avg H = {avg_entropy_for_this_k:.4f} (from {len(entropies_for_this_k)} runs)")
        print("Parallel simulations and aggregation finished.")
    else:
        print("No K values identified for running simulations based on the interval criteria.")

    # --- 4. Plotting K vs Entropy ---
    # (Plotting code remains the same as in the previous version)
    plt.figure(figsize=(14, 8))
    plt.plot(K_values_for_plot, theoretical_entropies_vs_K, 
             label=r'Theoretical $H(X_t | Y_n, \Delta_n)$', marker='.', linestyle='-')
    if simulated_K_points_aggregated:
        plt.plot(simulated_K_points_aggregated, simulated_entropies_aggregated, 
                 label=r'Avg. Simulated $H(X_t | Y_n, \Delta_n)$ (MC)', 
                 linestyle='none', marker='o', markersize=8, color='green')
    plt.axhline(y=h_x_source, color='r', linestyle=':', label=f'$H(X)$ (Source Entropy) = {h_x_source:.4f}')
    plt.xlabel("Number of Regions (K)")
    plt.ylabel("Entropy (bits)")
    plt.title(f"Conditional Entropy vs. Number of Regions (K)\n(MC runs: {num_mc_repetitions} per point, aggregated)")
    plt.legend()
    plt.grid(True)
    if K_values_for_plot:
        tick_step = max(1, (K_end_plot - K_start_plot) // 12 if K_end_plot > K_start_plot else 1)
        plt.xticks(np.arange(K_start_plot, K_end_plot + 1, step=tick_step))
    min_y_plot = 0; all_y_values_for_plot = []
    if theoretical_entropies_vs_K: all_y_values_for_plot.extend(theoretical_entropies_vs_K)
    if simulated_entropies_aggregated: all_y_values_for_plot.extend(simulated_entropies_aggregated)
    if h_x_source is not None: all_y_values_for_plot.append(h_x_source)
    valid_y_values_for_plot = [y for y in all_y_values_for_plot if y is not None and np.isfinite(y)]
    if valid_y_values_for_plot:
        max_y_plot_val = np.max(valid_y_values_for_plot)
        plt.ylim(bottom=min_y_plot, top=max_y_plot_val * 1.1 if max_y_plot_val > 0 else 1.0)
    elif h_x_source is not None: plt.ylim(bottom=min_y_plot, top=h_x_source * 1.1 if h_x_source > 0 else 1.0)
    else: plt.ylim(bottom=0, top=1.0) 
    plt.tight_layout()
    plt.show()

    print("\n--- End of K-sweep Plotting Script ---")