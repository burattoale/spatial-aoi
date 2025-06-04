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
                                 zeta = params.zeta,
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
                                 params.K, params.R_unit, params.alpha)
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

# Worker function for a single MC simulation task (defined globally or passed carefully)
def worker_mc_simulation_task(k_val_task, base_params_dict_task, beta_val_task, sim_runs_task, burn_in_task, mc_seed_task):
    current_sim_config_params = SimulationParameters(
        q=base_params_dict_task['q'], eta=base_params_dict_task['eta'],
        zeta=base_params_dict_task['zeta'], # Base zeta list
        epsilon=base_params_dict_task['epsilon'],
        rho=base_params_dict_task['rho'],
        m_override=base_params_dict_task['m_override'],
        K=k_val_task,
        alpha=base_params_dict_task['alpha'],
        beta=beta_val_task, # CURRENT BETA FOR THIS TASK
        R_unit=base_params_dict_task['R_unit'],
        X_symbols=base_params_dict_task['X_symbols']
    )

    avg_entropy_mc, _ = run_monte_carlo_simulation(
        params=current_sim_config_params,
        num_time_steps=sim_runs_task,
        num_burn_in_steps=burn_in_task,
        seed=mc_seed_task,
    )
    # Return k_val, beta_val, and entropy
    return k_val_task, beta_val_task, avg_entropy_mc


if __name__ == "__main__":
    # Define Simulation Parameters
    # zeta is now a list, NodeDistribution should be able to use it
    # If NodeDistribution expects a scalar zeta to generate a list of tx_probabilities,
    # then zeta here should be scalar, and NodeDistribution's logic adapted.
    # For now, assume NodeDistribution can use a list of base zetas, or the first element if it's a list of one.
    initial_params = SimulationParameters(
        q=0.005,
        eta=1,
        zeta=1e-2, # Example: list of base probabilities. NodeDistribution will use this.
        epsilon=0.1,
        rho=0.005,
        m_override=None,
        K=5, # Initial K for the first plot
        alpha=0.02,
        beta=0.02, # Default beta, will be overridden in the loop
        R_unit=10
    )

    # Monte Carlo settings for the first plot
    num_simulation_runs = 10000 # Reduced for faster example
    num_burn_in = 1000        # Reduced for faster example
    simulation_seed = 0

    print(f"Starting initial Monte Carlo simulation with parameters: {initial_params}")
    
    # For the first plot, we need NodeDistribution for the initial params to get tx_probs for sim
    _node_dist_initial = NodeDistribution(
        rho=initial_params.rho, unit_radius=initial_params.R_unit, K=initial_params.K,
        zeta=initial_params.zeta,
        alpha=initial_params.alpha, beta=initial_params.beta
    )
    m_calc_initial = len(_node_dist_initial)

    avg_entropy_mc_initial, entropy_time_series_initial = run_monte_carlo_simulation(
        initial_params,
        num_time_steps=num_simulation_runs,
        num_burn_in_steps=num_burn_in,
        seed=simulation_seed
    )
    print(f"\nInitial Monte Carlo Estimated Average H(X_t | Y_n, Delta_n): {avg_entropy_mc_initial:.4f}")

    dtmc_theoretical = DTMC(q=initial_params.q, eta=initial_params.eta, seed=simulation_seed)
    pi_source = dtmc_theoretical.pi
    h_x_source = -np.sum(pi_source * np.log2(pi_source, where=pi_source > 0))
    print(f"Entropy of source H(X): {h_x_source:.4f}")

    if m_calc_initial > 0:
        p_success_initial_k = ps_calc(m_calc_initial, _node_dist_initial.tx_probabilities, initial_params.epsilon)
        max_delta_initial_k = 1000#max(50, int(2 / (p_success_initial_k + 1e-9)))
        theoretical_entropy_initial = overall_entropy(
            A=dtmc_theoretical.A, pi=dtmc_theoretical.pi, K=initial_params.K,
            R_unit=initial_params.R_unit, alpha=initial_params.alpha, m=m_calc_initial,
            zeta=_node_dist_initial.tx_probabilities, epsilon=initial_params.epsilon,
            states=initial_params.X_symbols, max_delta_considered=max_delta_initial_k
        )
        print(f"Theoretical H(X_t | Y_n, Delta_n) (for K={initial_params.K}, beta={initial_params.beta}): {theoretical_entropy_initial:.4f}")
    else:
        theoretical_entropy_initial = h_x_source

    if list(entropy_time_series_initial):
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(len(entropy_time_series_initial)), entropy_time_series_initial, label=r'Instantaneous $H(X_t | Y_n, \Delta_n)$', alpha=0.7)
        plt.axhline(y=h_x_source, color='r', linestyle='--', label='$H(X)$')
        plt.axhline(y=avg_entropy_mc_initial, color='g', linestyle=':', label=r'Avg. Est. $H(MC)$')
        if m_calc_initial > 0:
            plt.axhline(y=theoretical_entropy_initial, color='purple', linestyle='-.', label=fr'Theoretical $H (K={initial_params.K})$')
        plt.xlabel(f"Time step (after {num_burn_in} burn-in)")
        plt.ylabel("Entropy (bits)")
        plt.title(rf"Evolution of Conditional Entropy (K={initial_params.K}, $\beta$={initial_params.beta})")
        plt.legend(); plt.grid(True); plt.ylim(bottom=0)
        if list(entropy_time_series_initial):
            max_plot_y = h_x_source * 1.1
            if avg_entropy_mc_initial is not None: max_plot_y = max(max_plot_y, avg_entropy_mc_initial * 1.1)
            if theoretical_entropy_initial is not None and m_calc_initial > 0: max_plot_y = max(max_plot_y, theoretical_entropy_initial * 1.1)
            if any(entropy_time_series_initial): max_plot_y = max(max_plot_y, np.max(entropy_time_series_initial) * 1.1)
            plt.ylim(top=max_plot_y)
        plt.tight_layout()
        plt.show()

    # --- K-sweep plot, now also iterating over BETA values ---
    print("\nGenerating plot for Entropy vs. K for different Beta values (with parallel simulations)...")

    beta_values_to_test = [0, 0.02,0.03,0.04]#, 4.0, 5, 6]  # Example Beta values

    K_start_plot = 1
    K_end_plot = 15
    K_values_for_plot = list(range(K_start_plot, K_end_plot + 1))
    
    simulation_K_interval = 1 
    num_mc_repetitions = 20 # Reduced for faster example with multiple betas
    n_parallel_jobs = -1      

    num_simulation_runs_sweep = 1000 # Reduced for faster example
    num_burn_in_sweep = 100          # Reduced for faster example

    # Store results per beta: {beta_val: {"K_points": [], "sim_H": [], "theo_H": []}}
    results_by_beta = defaultdict(lambda: {"K_points": [], "sim_H": [], "theo_H": []})

    for beta_val in beta_values_to_test:
        print(f"\n--- Processing for Beta = {beta_val} ---")
        
        current_beta_params = SimulationParameters( # Params object for this beta iteration
            q=initial_params.q, eta=initial_params.eta, zeta=initial_params.zeta, 
            epsilon=initial_params.epsilon, rho=initial_params.rho, 
            m_override=initial_params.m_override, K=initial_params.K, # K will be set per K-loop
            alpha=initial_params.alpha, beta=beta_val, # CURRENT BETA
            R_unit=initial_params.R_unit, X_symbols=initial_params.X_symbols
        )

        # 1. Theoretical Calculations for current beta (sequential for each K)
        theoretical_entropies_for_this_beta_vs_K = []
        print(f"Calculating Theoretical Entropies for Beta={beta_val}...")
        for k_val_theoretical in tqdm(K_values_for_plot, desc=f"Theoretical K-sweep (Beta={beta_val})"):
            _node_dist_k_theo = NodeDistribution(
                rho=current_beta_params.rho, unit_radius=current_beta_params.R_unit, 
                K=k_val_theoretical,
                zeta=current_beta_params.zeta, alpha=current_beta_params.alpha,
                beta=current_beta_params.beta, # Current beta
            )
            m_calc_k_theo = len(_node_dist_k_theo)
            
            if m_calc_k_theo > 0:
                ps_theo = ps_calc(m_calc_k_theo, _node_dist_k_theo.tx_probabilities, current_beta_params.epsilon)
                max_delta_k_theo = 1000#max(100, int(2 / (ps_theo + 1e-9))) # Adjusted min max_delta
                theoretical_H_k = overall_entropy(
                    A=dtmc_theoretical.A, pi=dtmc_theoretical.pi, K=k_val_theoretical, 
                    R_unit=current_beta_params.R_unit, alpha=current_beta_params.alpha, 
                    m=m_calc_k_theo, zeta=_node_dist_k_theo.tx_probabilities, 
                    epsilon=current_beta_params.epsilon, states=current_beta_params.X_symbols,
                    max_delta_considered=max_delta_k_theo
                )
                theoretical_entropies_for_this_beta_vs_K.append(theoretical_H_k)
            else:
                theoretical_entropies_for_this_beta_vs_K.append(h_x_source)
        results_by_beta[beta_val]["theo_H"] = theoretical_entropies_for_this_beta_vs_K
        results_by_beta[beta_val]["K_points"] = list(K_values_for_plot) # Store K points for theo line

        # 2. Determine K values requiring simulation runs (same for all betas, could be outside beta loop)
        k_values_to_simulate_explicitly = []
        if simulation_K_interval > 0:
            k_values_to_simulate_explicitly = [k for k in K_values_for_plot if (k - K_start_plot) % simulation_K_interval == 0]
            if K_start_plot not in k_values_to_simulate_explicitly: k_values_to_simulate_explicitly.insert(0, K_start_plot)
            if K_end_plot not in k_values_to_simulate_explicitly and (not k_values_to_simulate_explicitly or K_end_plot != k_values_to_simulate_explicitly[-1]):
                 k_values_to_simulate_explicitly.append(K_end_plot)
            k_values_to_simulate_explicitly = sorted(list(set(k_values_to_simulate_explicitly)))
        elif K_values_for_plot:
            k_values_to_simulate_explicitly.append(K_start_plot)
            if K_end_plot != K_start_plot: k_values_to_simulate_explicitly.append(K_end_plot)
            k_values_to_simulate_explicitly = sorted(list(set(k_values_to_simulate_explicitly)))

        # 3. Parallel Monte Carlo Simulations for current beta
        if k_values_to_simulate_explicitly:
            print(f"Preparing Parallel MC Simulations for Beta={beta_val}...")
            
            base_params_for_worker = { # Dictionary for parameters constant across K for THIS beta
                'q': current_beta_params.q, 'eta': current_beta_params.eta, 'zeta': current_beta_params.zeta,
                'epsilon': current_beta_params.epsilon, 'rho': current_beta_params.rho,
                'm_override': current_beta_params.m_override, 'alpha': current_beta_params.alpha,
                'R_unit': current_beta_params.R_unit, # Beta is passed directly to worker
                'X_symbols': current_beta_params.X_symbols
            }

            simulation_job_args_list = []
            for k_to_sim in k_values_to_simulate_explicitly:
                for rep_idx in range(num_mc_repetitions):
                    # Unique seed: incorporates K, beta (as int for simplicity), and repetition
                    unique_run_seed = simulation_seed + (k_to_sim * 10000) + (int(beta_val * 100)) + rep_idx 
                    simulation_job_args_list.append(
                        (k_to_sim, base_params_for_worker, beta_val, num_simulation_runs_sweep, num_burn_in_sweep, unique_run_seed)
                    )
            
            print(f"Submitting {len(simulation_job_args_list)} MC jobs for Beta={beta_val} ({n_parallel_jobs} cores)...")
            
            parallel_results_list = Parallel(n_jobs=n_parallel_jobs, verbose=1)( # Reduced verbosity for multiple betas
                delayed(worker_mc_simulation_task)(*args) for args in tqdm(simulation_job_args_list, desc=f"MC Jobs (Beta={beta_val})")
            )

            mc_results_by_k_map_this_beta = defaultdict(list)
            for k_r, beta_r, entropy_r in parallel_results_list: # Worker now returns beta too
                if beta_r == beta_val: # Ensure we are aggregating for the correct beta
                    mc_results_by_k_map_this_beta[k_r].append(entropy_r)

            current_beta_sim_K = []
            current_beta_sim_H_avg = []
            sorted_sim_k_values_this_beta = sorted(mc_results_by_k_map_this_beta.keys())

            for k_agg in sorted_sim_k_values_this_beta:
                avg_H = np.mean(mc_results_by_k_map_this_beta[k_agg])
                current_beta_sim_K.append(k_agg)
                current_beta_sim_H_avg.append(avg_H)
                # print(f"Aggregated MC for Beta={beta_val}, K={k_agg}: Avg H = {avg_H:.4f}")
            
            results_by_beta[beta_val]["sim_K_points"] = current_beta_sim_K
            results_by_beta[beta_val]["sim_H"] = current_beta_sim_H_avg
            print(f"Parallel simulations for Beta={beta_val} finished.")
        else:
            print(f"No K values for MC simulation for Beta={beta_val}.")

    # --- 4. Plotting K vs Entropy for different Betas ---
    plt.figure(figsize=(14, 8))
    
    # Define a list of colors for different beta lines, or use a colormap
    beta_colors = plt.cm.viridis(np.linspace(0, 0.9, len(beta_values_to_test)))

    for i, beta_val_plot in enumerate(beta_values_to_test):
        beta_data = results_by_beta[beta_val_plot]
        color = beta_colors[i]

        # Plot Theoretical Line for this Beta (optional, can make plot busy)
        if beta_data["K_points"] and beta_data["theo_H"]:
            plt.semilogy(beta_data["K_points"], beta_data["theo_H"],
                     label=fr'Theoretical $H (\beta={beta_val_plot:.2f})$',
                     marker='.', linestyle='--', color=color, alpha=0.7)
        
        # Plot Averaged Simulated Entropies for this Beta
        if beta_data.get("sim_K_points") and beta_data.get("sim_H"): # Use .get for safety
            plt.semilogy(beta_data["sim_K_points"], beta_data["sim_H"], 
                     label=fr'Avg. Simulated $H (MC, \beta={beta_val_plot:.2f})$', 
                     linestyle='-', marker='o', markersize=6, color=color)

    plt.axhline(y=h_x_source, color='r', linestyle=':', label=f'$H(X)$ = {h_x_source:.4f}')
    
    plt.xlabel("Number of Regions (K)")
    plt.ylabel("Entropy (bits)")
    plt.title(f"Conditional Entropy vs. K for different $\\beta$ values\n(MC runs: {num_mc_repetitions} per point, aggregated)")
    plt.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    if K_values_for_plot:
        tick_step = max(1, (K_end_plot - K_start_plot) // 12 if K_end_plot > K_start_plot else 1)
        plt.xticks(np.arange(K_start_plot, K_end_plot + 1, step=tick_step))
    
    min_y_plot = 0
    all_y_values_for_plot = [h_x_source] if h_x_source is not None else []
    for beta_val_plot in beta_values_to_test:
        beta_data = results_by_beta[beta_val_plot]
        if beta_data["theo_H"]: all_y_values_for_plot.extend(beta_data["theo_H"])
        if beta_data.get("sim_H"): all_y_values_for_plot.extend(beta_data.get("sim_H"))
    
    valid_y_values_for_plot = [y for y in all_y_values_for_plot if y is not None and np.isfinite(y)]
    if valid_y_values_for_plot:
        max_y_plot_val = np.max(valid_y_values_for_plot)
        plt.ylim(bottom=min_y_plot, top=max_y_plot_val * 1.1 if max_y_plot_val > 0 else 1.0)
    elif h_x_source is not None: plt.ylim(bottom=min_y_plot, top=h_x_source * 1.1 if h_x_source > 0 else 1.0)
    else: plt.ylim(bottom=0, top=1.0) 
        
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust rect to make space for legend outside
    plt.show()

    print("\n--- End of Script ---")