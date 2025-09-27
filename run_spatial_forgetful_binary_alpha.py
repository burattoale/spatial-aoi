#!/usr/bin/env python3
"""
Spatial-aware forgetful model simulations for binary_alpha_loc_aware scenario
Runs simulations for alpha values: 0.02, 0.06, 0.1
Compatible with the binary_source_loc_aware_alpha.pkl data structure
"""

import numpy as np
import math
import pickle
import os
import time
from copy import deepcopy
from joblib import Parallel, delayed
from tqdm import tqdm

from utils import SimulationParameters
from utils.forgetful_mc_simulations import run_monte_carlo_simulation_spatial

def run_single_spatial_simulation(params, num_steps, burn_in, seed, num_topologies=20):
    """Run spatial-aware forgetful simulation for multiple topologies."""
    entropies = []
    
    for topo_idx in range(num_topologies):
        # Use different seed for each topology
        topo_seed = seed + topo_idx * 1000
        
        try:
            avg_entropy, _ = run_monte_carlo_simulation_spatial(
                params, 
                num_steps, 
                burn_in, 
                seed=topo_seed,
                discard_logic=False
            )
            entropies.append(avg_entropy)
        except Exception as e:
            print(f"Error in simulation (seed {topo_seed}): {e}")
            entropies.append(np.nan)
    
    # Filter out NaN values
    valid_entropies = [h for h in entropies if not np.isnan(h)]
    
    if valid_entropies:
        return {
            'mean': np.mean(valid_entropies),
            'std': np.std(valid_entropies),
            'count': len(valid_entropies),
            'raw_values': valid_entropies
        }
    else:
        return {
            'mean': np.nan,
            'std': np.nan,
            'count': 0,
            'raw_values': []
        }

def run_spatial_forgetful_experiments(alpha_values, n_jobs=4):
    """Run spatial-aware forgetful experiments for multiple alpha values."""
    
    # Base parameters matching the binary source scenario
    base_params = SimulationParameters(
        q=0.005,
        eta=1,
        zeta=1e-4,
        epsilon=0.1,
        rho=0.05,
        alpha=0.02,  # Will be overridden
        beta=0,
        R_unit=5,  # Unit radius
        X_symbols=[0, 1],
        Y_symbols=[0, 1, 2]
    )
    
    # K range similar to binary_source_loc_aware_alpha
    K_values = list(range(1, 65))  # K from 1 to 64
    
    # Simulation parameters
    num_steps = 15000
    burn_in = 1000
    num_topologies = 20
    
    results = {}
    
    print(f"Running spatial-aware forgetful simulations...")
    print(f"Alpha values: {alpha_values}")
    print(f"K range: {K_values[0]} to {K_values[-1]}")
    print(f"Simulation steps: {num_steps}, Burn-in: {burn_in}")
    print(f"Topologies per (α, K): {num_topologies}")
    print(f"Total simulations: {len(alpha_values)} × {len(K_values)} × {num_topologies} = {len(alpha_values) * len(K_values) * num_topologies}")
    
    for alpha in alpha_values:
        print(f"\n=== Processing α = {alpha} ===")
        
        # Create parameter set for this alpha
        alpha_params = deepcopy(base_params)
        alpha_params.alpha = alpha
        
        # Prepare simulation tasks for this alpha
        tasks = []
        for K in K_values:
            params = deepcopy(alpha_params)
            params.K = K
            
            # Calculate number of nodes
            if params.m_override is None:
                params.m = math.floor(params.rho * np.pi * (params.K * params.R_unit) ** 2)
            else:
                params.m = params.m_override
            
            # Create unique seed for this combination
            seed = hash((alpha, K)) % (2**32 - 1)
            
            tasks.append((params, num_steps, burn_in, seed, num_topologies, K))
        
        # Run simulations in parallel
        print(f"  Running {len(tasks)} simulations (K={K_values[0]} to {K_values[-1]})...")
        parallel_results = Parallel(n_jobs=n_jobs)(
            delayed(lambda task: (task[5], run_single_spatial_simulation(task[0], task[1], task[2], task[3], task[4])))(task)
            for task in tqdm(tasks, desc=f"α={alpha}")
        )
        
        # Organize results
        alpha_results = {}
        for K, result in parallel_results:
            alpha_results[K] = result
        
        results[f'alpha:{alpha}'] = alpha_results
        
        # Print summary for this alpha
        valid_results = [res for res in alpha_results.values() if res['count'] > 0]
        if valid_results:
            mean_entropies = [res['mean'] for res in valid_results]
            print(f"  Completed simulations: {len(valid_results)}/{len(K_values)}")
            print(f"  Entropy range: [{np.min(mean_entropies):.6f}, {np.max(mean_entropies):.6f}]")
            print(f"  Average entropy: {np.mean(mean_entropies):.6f}")
        else:
            print(f"  No valid results for α={alpha}")
    
    return results

def save_results(results, filename):
    """Save results in a format compatible with binary_source_loc_aware_alpha.pkl"""
    
    # Create metadata
    metadata = {
        'script_name': __file__,
        'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Spatial-aware forgetful model simulations for binary alpha scenario',
        'simulation_parameters': {
            'num_steps': 15000,
            'burn_in': 1000,
            'num_topologies': 20,
            'K_range': [1, 64],
            'alpha_values': [0.02, 0.06, 0.1]
        }
    }
    
    # Structure data similar to existing format
    # The existing binary_source_loc_aware_alpha.pkl has structure:
    # {
    #   'hmm': {'alpha:0.02': {K: entropy_value, ...}, ...},
    #   'forgetful': {...},  # This is what we're creating
    #   'hmm_err': {...}
    # }
    
    # Try to load existing data to preserve HMM results
    output_data = {'forgetful_spatial': results}
    
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                existing_data = pickle.load(f)
            
            # Preserve existing data
            output_data.update(existing_data)
            
            # Add our results
            output_data['forgetful_spatial'] = results
            
            print(f"Updated existing file with spatial forgetful results")
            
        except Exception as e:
            print(f"Warning: Could not load existing file: {e}")
            print(f"Creating new file with only spatial forgetful results")
    
    # Add metadata
    output_data['metadata_spatial'] = metadata
    
    # Save results
    with open(filename, 'wb') as f:
        pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Results saved to: {filename}")

def create_summary_report(results):
    """Create a summary report of the simulation results."""
    
    print("\n" + "="*80)
    print("SPATIAL-AWARE FORGETFUL SIMULATION SUMMARY")
    print("="*80)
    
    for alpha_key, alpha_data in results.items():
        alpha_val = float(alpha_key.split(':')[1])
        print(f"\nα = {alpha_val}:")
        
        # Collect statistics
        valid_results = [(K, res) for K, res in alpha_data.items() if res['count'] > 0]
        
        if valid_results:
            K_values = [K for K, _ in valid_results]
            mean_entropies = [res['mean'] for _, res in valid_results]
            std_entropies = [res['std'] for _, res in valid_results]
            
            print(f"  Valid simulations: {len(valid_results)}/{len(alpha_data)}")
            print(f"  K range: {min(K_values)} - {max(K_values)}")
            print(f"  Entropy statistics:")
            print(f"    Mean: {np.mean(mean_entropies):.6f} ± {np.std(mean_entropies):.6f}")
            print(f"    Range: [{np.min(mean_entropies):.6f}, {np.max(mean_entropies):.6f}]")
            print(f"    Std range: [{np.min(std_entropies):.6f}, {np.max(std_entropies):.6f}]")
            
            # Find minimum entropy and optimal K
            min_idx = np.argmin(mean_entropies)
            min_K = K_values[min_idx]
            min_entropy = mean_entropies[min_idx]
            print(f"  Optimal K: {min_K} (entropy = {min_entropy:.6f})")
            
        else:
            print(f"  No valid results")

def main():
    """Main function to run spatial-aware forgetful simulations."""
    
    # Alpha values for binary_alpha_loc_aware scenario
    alpha_values = [0.02, 0.06, 0.1]
    
    # Number of parallel jobs
    n_jobs = 3
    
    print("="*80)
    print("SPATIAL-AWARE FORGETFUL MODEL SIMULATIONS")
    print("Binary Alpha Location-Aware Scenario")
    print("="*80)
    
    # Run simulations
    results = run_spatial_forgetful_experiments(alpha_values, n_jobs=n_jobs)
    
    # Create summary report
    create_summary_report(results)
    
    # Save results
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as separate file first
    spatial_filename = os.path.join(output_dir, 'binary_source_loc_aware_alpha_spatial_forgetful.pkl')
    save_results(results, spatial_filename)
    
    # Also try to update the main binary_source_loc_aware_alpha.pkl file
    main_filename = os.path.join(output_dir, 'binary_source_loc_aware_alpha.pkl')
    try:
        save_results(results, main_filename)
        print(f"Successfully updated main results file: {main_filename}")
    except Exception as e:
        print(f"Could not update main file: {e}")
        print(f"Results saved separately as: {spatial_filename}")
    
    print(f"\n🎉 Spatial-aware forgetful simulations completed!")
    print(f"📊 Results available for α = {alpha_values}")
    print(f"📁 Output files:")
    print(f"   - {spatial_filename}")
    print(f"   - {main_filename} (if update successful)")

if __name__ == "__main__":
    main()