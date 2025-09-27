#!/usr/bin/env python3
"""
Compute and compare CDF of entropy for HMM and Forgetful models with different K values.
Unit radius = 5
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from hmm_joint_prob import hmm_entropy
from utils import SimulationParameters
from utils.stif_h_agnostic_python import stif_h_agnostic_new

def compute_hmm_entropy_cdf_comparison():
    """
    Compute CDF of entropy for HMM and Forgetful models with:
    - alpha = 0.02
    - 2 K values: 5 and 25
    - Unit radius = 5
    """
    # Base configuration
    sim_length = 1000000
    q = 0.005
    eta = 1
    zeta = 1e-4
    epsilon = 0.1
    rho = 5e-2
    R_unit = 5  # Unit radius = 5 as requested
    x_symbols = [0, 1]
    y_symbols = [0, 1, 2]
    
    # Parameter combinations to test
    alpha_values = [0.02]
    K_values = [5, 25]
    
    # Storage for results
    results = {}
    
    print("Computing HMM and Forgetful entropy CDF for different parameter combinations...")
    print(f"Unit radius: {R_unit}")
    print(f"Simulation length: {sim_length}")
    print(f"Alpha values: {alpha_values}")
    print(f"K values: {K_values}")
    print()
    
    # Iterate through all combinations
    for alpha in alpha_values:
        results[alpha] = {}
        for K in K_values:
            print(f"--- Processing alpha={alpha}, K={K} ---")
            
            # Calculate number of nodes
            m = math.floor(rho * np.pi * (R_unit * K)**2)
            
            # Create parameters
            params = SimulationParameters(
                q=q, eta=eta, zeta=zeta, epsilon=epsilon, 
                rho=rho, m=m, K=K, alpha=alpha, R_unit=R_unit,
                X_symbols=x_symbols, Y_symbols=y_symbols
            )
            
            print(f"  Number of sensors (m): {m}")
            
            # Run HMM entropy simulation
            print("  Running HMM simulation...")
            entropies, mean_entropy, est_error, short_est_prob, data_tuple, cdf_tuple = hmm_entropy(
                params, simulation_length=sim_length, seed=42  # Fixed seed for reproducibility
            )
            
            # Unpack HMM results
            Y, X_true, cond_probs_x1, cond_probs_x = data_tuple
            cdf_h_hmm, gamma_vec_hmm = cdf_tuple
            
            # Run Forgetful model computation
            print("  Running Forgetful (STIF) computation...")
            H_stif, cdf_h_stif, gamma_vec_stif = stif_h_agnostic_new(
                R=R_unit, K=K, alpha=alpha, rho=rho, zeta=zeta, q=q, eta=eta, epsilon=epsilon
            )
            
            # Store results
            results[alpha][K] = {
                'hmm': {
                    'entropies': entropies,
                    'mean_entropy': mean_entropy,
                    'est_error': est_error,
                    'cdf_h': cdf_h_hmm,
                    'gamma_vec': gamma_vec_hmm,
                },
                'forgetful': {
                    'mean_entropy': H_stif,
                    'cdf_h': cdf_h_stif,
                    'gamma_vec': gamma_vec_stif,
                },
                'm': m
            }
            
            print(f"  HMM Mean entropy: {mean_entropy:.6f}")
            print(f"  Forgetful Mean entropy: {H_stif:.6f}")
            print()

    # Export TikZ data to text files
    import os
    tikz_data_dir = 'plots/cdf_entropy'
    os.makedirs(tikz_data_dir, exist_ok=True)
    
    print("Exporting TikZ data to text files...")
    
    # Export data for each parameter combination
    for alpha in alpha_values:
        for K in K_values:
            result = results[alpha][K]
            
            # HMM data
            filename_hmm = f"cdf_hmm_alpha{str(alpha).replace('.', '')}_K{K}.txt"
            filepath_hmm = os.path.join(tikz_data_dir, filename_hmm)
            with open(filepath_hmm, 'w') as f:
                f.write(f"% HMM CDF data for alpha={alpha}, K={K}\n")
                f.write(f"% Columns: entropy_threshold cdf_value\n")
                for x, y in zip(result['hmm']['gamma_vec'], result['hmm']['cdf_h']):
                    f.write(f"{x:.6f}\t{y:.6f}\n")
            print(f"  ✓ Exported {filename_hmm}")

            # Forgetful data
            filename_forgetful = f"cdf_forgetful_alpha{str(alpha).replace('.', '')}_K{K}.txt"
            filepath_forgetful = os.path.join(tikz_data_dir, filename_forgetful)
            with open(filepath_forgetful, 'w') as f:
                f.write(f"% Forgetful CDF data for alpha={alpha}, K={K}\n")
                f.write(f"% Columns: entropy_threshold cdf_value\n")
                for x, y in zip(result['forgetful']['gamma_vec'], result['forgetful']['cdf_h']):
                    f.write(f"{x:.6f}\t{y:.6f}\n")
            print(f"  ✓ Exported {filename_forgetful}")

    # Create single CDF plot with log y-scale
    plt.figure(figsize=(12, 8))
    
    # Colors and styles for different combinations
    colors = {
        'hmm_5': 'blue',
        'hmm_25': 'red', 
        'forgetful_5': 'green',
        'forgetful_25': 'orange'
    }
    
    linestyles = {
        'hmm_5': '-',
        'hmm_25': '--', 
        'forgetful_5': '-.',
        'forgetful_25': ':'
    }
    
    # Plot regular CDF with log y-scale
    for alpha in alpha_values:
        for K in K_values:
            radius = R_unit * K;
            
            # HMM Plot
            result_hmm = results[alpha][K]['hmm']
            cdf_hmm = np.maximum(result_hmm['cdf_h'], 1e-6)
            plt.semilogy(result_hmm['gamma_vec'], cdf_hmm,
                        color=colors[f'hmm_{K}'], 
                        linestyle=linestyles[f'hmm_{K}'],
                        linewidth=2.5,
                        label=f'HMM, R={radius}',
                        marker='', markersize=0)

            # Forgetful Plot
            result_forgetful = results[alpha][K]['forgetful']
            cdf_forgetful = np.maximum(result_forgetful['cdf_h'], 1e-6)
            plt.semilogy(result_forgetful['gamma_vec'], cdf_forgetful,
                        color=colors[f'forgetful_{K}'], 
                        linestyle=linestyles[f'forgetful_{K}'],
                        linewidth=2.5,
                        label=f'Forgetful, R={radius}',
                        marker='', markersize=0)

    plt.xlabel('Entropy threshold $\gamma$', fontsize=14)
    plt.ylabel('$P(H \leq \gamma)$', fontsize=14)
    plt.title('CDF of Entropy for HMM and Forgetful Models (Log scale)', fontsize=16)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, which='both')
    plt.xlim(0, 1)
    plt.ylim(1e-6, 1)
    
    # Improve tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('hmm_vs_forgetful_entropy_cdf_logy.png', dpi=300, bbox_inches='tight')
    plt.savefig('hmm_vs_forgetful_entropy_cdf_logy.pdf', bbox_inches='tight')
    print("✓ CDF plot with log y-scale saved to hmm_vs_forgetful_entropy_cdf_logy.png and .pdf")
    
    # Export combined TikZ data file for the plot
    combined_tikz_file = os.path.join(tikz_data_dir, 'cdf_combined_logy.txt')
    with open(combined_tikz_file, 'w') as f:
        f.write("% Combined CDF data for TikZ plotting (log y-scale)\n")
        f.write("% Format: entropy_threshold cdf_hmm_K5 cdf_hmm_K25 cdf_forgetful_K5 cdf_forgetful_K25\n")
        
        # Use HMM K=5 gamma vector as reference, assuming they are similar
        ref_gamma = results[alpha_values[0]][K_values[0]]['hmm']['gamma_vec']
        
        for i, gamma in enumerate(ref_gamma):
            f.write(f"{gamma:.6f}")
            
            # Interpolate and write CDF values for each model and K
            for model in ['hmm', 'forgetful']:
                for K in K_values:
                    result = results[alpha_values[0]][K][model]
                    # Interpolate to get the cdf value at the reference gamma
                    cdf_val = np.interp(gamma, result['gamma_vec'], result['cdf_h'])
                    cdf_val = max(cdf_val, 1e-6)  # Avoid log(0)
                    f.write(f"\t{cdf_val:.6e}")
            f.write("\n")
    
    print(f"✓ Combined TikZ data exported to {combined_tikz_file}")
    
    # Create TikZ template file
    tikz_template_file = os.path.join(tikz_data_dir, 'cdf_plot_template.tex')
    with open(tikz_template_file, 'w') as f:
        f.write("""% TikZ template for HMM vs Forgetful CDF plot with log y-scale\n\begin{tikzpicture}\n\begin{semilogyaxis}[\n    xlabel={Entropy threshold $\gamma$},\n    ylabel={$P(H \leq \gamma)$},\n    title={CDF of Entropy: HMM vs. Forgetful},\n    legend pos=south east,\n    grid=both,\n    grid style={line width=.1pt, draw=gray!10},\n    major grid style={line width=.2pt,draw=gray!50},\n    xmin=0, xmax=1,\n    ymin=1e-6, ymax=1,\n    width=12cm, height=9cm\n]\n\n% Data plots\n% Columns: 0=gamma, 1=hmm_k5, 2=hmm_k25, 3=forgetful_k5, 4=forgetful_k25\n\addplot[blue, line width=1.5pt, solid] table [x index=0, y index=1] {cdf_combined_logy.txt};\n\addplot[red, line width=1.5pt, dashed] table [x index=0, y index=2] {cdf_combined_logy.txt};\n\addplot[green, line width=1.5pt, dashdotted] table [x index=0, y index=3] {cdf_combined_logy.txt};\n\addplot[orange, line width=1.5pt, dotted] table [x index=0, y index=4] {cdf_combined_logy.txt};\n\n% Legend\n\legend{HMM, R=25, HMM, R=125, Forgetful, R=25, Forgetful, R=125}\n\end{semilogyaxis}\n\end{tikzpicture}\n""")
    
    print(f"✓ TikZ template saved to {tikz_template_file}")
    
    # Print detailed analysis
    print("\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)
    
    for alpha in alpha_values:
        print(f"\nAlpha = {alpha}:")
        for K in K_values:
            radius = R_unit * K
            result = results[alpha][K]
            
            print(f"  K = {K} (R = {radius}, m = {result['m']} sensors):")
            
            # HMM Analysis
            hmm_res = result['hmm']
            entropies = hmm_res['entropies']
            percentiles = [25, 50, 75, 90, 95]
            p_values = np.percentile(entropies, percentiles)
            print(f"    HMM Model:")
            print(f"      Mean entropy: {hmm_res['mean_entropy']:.6f}")
            print(f"      Std entropy:  {np.std(entropies):.6f}")
            print(f"      Percentiles:  " + ", ".join([f"{p}%={v:.3f}" for p, v in zip(percentiles, p_values)]))
            print(f"      Est. error:   {hmm_res['est_error']:.6f}")

            # Forgetful Analysis
            forgetful_res = result['forgetful']
            print(f"    Forgetful Model:")
            print(f"      Mean entropy: {forgetful_res['mean_entropy']:.6f}")

    # Cross-parameter analysis
    print(f"\n" + "-"*40)
    print("CROSS-PARAMETER ANALYSIS (HMM):")
    print("-"*40)
    
    print(f"Effect of increasing K (sectors) for HMM:")
    for alpha in alpha_values:
        mean_K5 = results[alpha][5]['hmm']['mean_entropy']
        mean_K25 = results[alpha][25]['hmm']['mean_entropy']
        change = ((mean_K25 - mean_K5) / mean_K5) * 100
        print(f"  α={alpha}: K=5→K=25 changes HMM mean entropy by {change:+.1f}%")

    print(f"\n" + "-"*40)
    print("CROSS-PARAMETER ANALYSIS (Forgetful):")
    print("-"*40)
    
    print(f"Effect of increasing K (sectors) for Forgetful:")
    for alpha in alpha_values:
        mean_K5 = results[alpha][5]['forgetful']['mean_entropy']
        mean_K25 = results[alpha][25]['forgetful']['mean_entropy']
        change = ((mean_K25 - mean_K5) / mean_K5) * 100
        print(f"  α={alpha}: K=5→K=25 changes Forgetful mean entropy by {change:+.1f}%")

    return results

if __name__ == "__main__":
    results = compute_hmm_entropy_cdf_comparison()
    print(f"\n🎉 Analysis completed successfully!")
    print(f"📊 CDF plot comparing HMM and Forgetful models generated.")
    print(f"📁 Check 'hmm_vs_forgetful_entropy_cdf_logy.png' for the plot.")
    print(f"📊 TikZ data files exported to 'tikz_data/' directory")

