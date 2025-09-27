#!/usr/bin/env python3
"""
Plot comprehensive results from forgetful_binary_spawc_complete.pkl
Creates plots for:
1. Entropy vs Radius for eta=1 and eta=5 (analytical lines + MC points)
2. Minimum entropy vs eta for both alpha values
3. Optimal radius vs eta for both alpha values
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from matplotlib.patches import Rectangle

def load_results():
    """Load the forgetful binary spawc results."""
    with open('results/forgetful_binary_spawc_complete.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def extract_analytical_data(results, alpha, eta_val):
    """Extract analytical data for specific alpha and eta."""
    alpha_key = f'alpha_{alpha}'
    eta_key = f'eta_{eta_val}'
    
    if alpha_key not in results['results']:
        return None, None
    
    alpha_data = results['results'][alpha_key]
    
    if eta_key not in alpha_data['analytical']:
        return None, None
    
    analytical_data = alpha_data['analytical'][eta_key]
    K_values = sorted(analytical_data.keys())
    entropies = [analytical_data[K] for K in K_values]
    radii = [K * alpha_data['base_params'].R_unit for K in K_values]
    
    return radii, entropies

def extract_mc_data(results, alpha, eta_val):
    """Extract Monte Carlo data for specific alpha and eta."""
    alpha_key = f'alpha_{alpha}'
    eta_key = f'eta_{eta_val}'
    
    if alpha_key not in results['results']:
        return None, None, None
    
    alpha_data = results['results'][alpha_key]
    
    if eta_key not in alpha_data['monte_carlo']:
        return None, None, None
    
    mc_data = alpha_data['monte_carlo'][eta_key]
    K_values = sorted(mc_data.keys())
    means = []
    stds = []
    radii = []
    
    for K in K_values:
        if 'mean' in mc_data[K] and not np.isnan(mc_data[K]['mean']):
            means.append(mc_data[K]['mean'])
            stds.append(mc_data[K]['std'])
            radii.append(K * alpha_data['base_params'].R_unit)
    
    return radii, means, stds

def find_minimum_entropy_over_eta(results, alpha):
    """Find minimum entropy and optimal radius for each eta value."""
    alpha_key = f'alpha_{alpha}'
    alpha_data = results['results'][alpha_key]
    
    eta_values = []
    min_entropies = []
    optimal_radii = []
    optimal_K_values = []
    
    # Process all eta values in increasing order
    eta_keys_sorted = []
    for eta_key in alpha_data['analytical'].keys():
        if eta_key.startswith('eta_'):
            eta_val = float(eta_key.replace('eta_', ''))
            eta_keys_sorted.append((eta_val, eta_key))
    
    # Sort by eta value
    eta_keys_sorted.sort(key=lambda x: x[0])
    
    for eta_val, eta_key in eta_keys_sorted:
        eta_values.append(eta_val)
        
        # Get analytical data for this eta
        analytical_data = alpha_data['analytical'][eta_key]
        K_values = sorted(analytical_data.keys())
        
        # Discard the first 10 values of K for finding minimum
        K_values_filtered = [K for K in K_values if K > 10]
        entropies = [analytical_data[K] for K in K_values_filtered if not np.isnan(analytical_data[K])]
        valid_K_values = [K for K in K_values_filtered if not np.isnan(analytical_data[K])]
        
        if entropies:
            min_idx = np.argmin(entropies)
            min_entropy = entropies[min_idx]
            optimal_K = valid_K_values[min_idx]
            optimal_R = optimal_K * alpha_data['base_params'].R_unit
            
            min_entropies.append(min_entropy)
            optimal_radii.append(optimal_R)
            optimal_K_values.append(optimal_K)
    
    return eta_values, min_entropies, optimal_radii, optimal_K_values
    
    return eta_values, min_entropies, optimal_radii, optimal_K_values

def create_entropy_vs_radius_plots(results):
    """Create entropy vs radius plot for α=0.02 with η=1 and η=5, starting from K=2."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    alpha = 0.02  # Only use α=0.02
    eta_values = [1.0, 5.0]
    colors = {1.0: 'blue', 5.0: 'red'}
    linestyles = {1.0: '-', 5.0: '--'}
    
    # Create TikZ data directory
    tikz_dir = 'plots/tikz_entropy_vs_radius'
    os.makedirs(tikz_dir, exist_ok=True)
    
    plot_data = {}
    
    for eta in eta_values:
        # Plot analytical lines (starting from K=2)
        radii_anal, entropies_anal = extract_analytical_data(results, alpha, eta)
        if radii_anal is not None:
            # Filter to start from K=2 (radius = 2*5 = 10)
            filtered_data = [(r, h) for r, h in zip(radii_anal, entropies_anal) if r >= 10]
            if filtered_data:
                radii_filtered, entropies_filtered = zip(*filtered_data)
                ax.plot(radii_filtered, entropies_filtered, 
                       color=colors[eta], linewidth=2.5, 
                       label=f'Analytical η={eta}', linestyle=linestyles[eta])
                
                # Store data for TikZ
                plot_data[f'analytical_eta_{eta}'] = list(zip(radii_filtered, entropies_filtered))
        
        # Plot MC points (averages only, no error bars)
        radii_mc, means_mc, stds_mc = extract_mc_data(results, alpha, eta)
        if radii_mc is not None:
            # Filter MC data to start from K=2 as well
            filtered_mc = [(r, m) for r, m in zip(radii_mc, means_mc) if r >= 10]
            if filtered_mc:
                radii_mc_filtered, means_mc_filtered = zip(*filtered_mc)
                ax.plot(radii_mc_filtered, means_mc_filtered,
                       color=colors[eta], marker='o', markersize=6,
                       linestyle='', label=f'MC η={eta}', alpha=0.8)
                
                # Store MC data for TikZ
                plot_data[f'mc_eta_{eta}'] = list(zip(radii_mc_filtered, means_mc_filtered))
    
    ax.set_xlabel('Coverage Radius R', fontsize=14)
    ax.set_ylabel('Entropy H (bits)', fontsize=14)
    ax.set_title(f'Entropy vs Radius (α = {alpha})', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig('plots/entropy_vs_radius.png', dpi=300, bbox_inches='tight')
    plt.savefig('plots/entropy_vs_radius.pdf', bbox_inches='tight')
    
    # Export TikZ data files
    for data_type, data_points in plot_data.items():
        filename = f'{tikz_dir}/{data_type}.txt'
        with open(filename, 'w') as f:
            f.write(f'% {data_type} data for entropy vs radius plot\n')
            f.write('% Columns: radius entropy\n')
            for radius, entropy in data_points:
                f.write(f'{radius:.2f}\t{entropy:.6f}\n')
    
    # Create TikZ template
    template_file = f'{tikz_dir}/plot_template.tex'
    with open(template_file, 'w') as f:
        f.write("""% TikZ template for entropy vs radius plot
\\begin{tikzpicture}
\\begin{axis}[
    xlabel={Coverage Radius $R$},
    ylabel={Entropy $H$ (bits)},
    title={Entropy vs Radius ($\\alpha = 0.02$)},
    legend pos=north east,
    grid=both,
    grid style={line width=.1pt, draw=gray!10},
    major grid style={line width=.2pt,draw=gray!50},
    width=12cm, height=9cm
]

% Analytical curves
\\addplot[blue, line width=1.5pt, solid] table {analytical_eta_1.0.txt};
\\addplot[red, line width=1.5pt, dashed] table {analytical_eta_5.0.txt};

% MC points
\\addplot[blue, only marks, mark=o, mark size=2pt] table {mc_eta_1.0.txt};
\\addplot[red, only marks, mark=square, mark size=2pt] table {mc_eta_5.0.txt};

\\legend{Analytical $\\eta=1$, Analytical $\\eta=5$, MC $\\eta=1$, MC $\\eta=5$}

\\end{axis}
\\end{tikzpicture}
""")
    
    print(f"  ✓ TikZ data exported to {tikz_dir}/")
    
    return fig

def create_minimum_entropy_plot(results):
    """Create minimum entropy vs eta plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    alpha_values = [0.02, 0.06]
    colors = {'0.02': 'blue', '0.06': 'red'}
    
    # Create TikZ data directory
    tikz_dir = 'plots/tikz_minimum_entropy'
    os.makedirs(tikz_dir, exist_ok=True)
    
    plot_data = {}
    
    for alpha in alpha_values:
        eta_vals, min_entropies, opt_radii, opt_K = find_minimum_entropy_over_eta(results, alpha)
        
        ax.plot(eta_vals, min_entropies, 
                color=colors[str(alpha)], marker='o', linewidth=2.5, markersize=6,
                label=f'α = {alpha}')
        
        # Store data for TikZ
        plot_data[f'alpha_{str(alpha).replace(".", "")}'] = list(zip(eta_vals, min_entropies))
    
    ax.set_xlabel('η (Markov chain parameter)', fontsize=14)
    ax.set_ylabel('Minimum Entropy H (bits)', fontsize=14)
    ax.set_title('Minimum Entropy vs η', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig('plots/minimum_entropy_vs_eta.png', dpi=300, bbox_inches='tight')
    plt.savefig('plots/minimum_entropy_vs_eta.pdf', bbox_inches='tight')
    
    # Export TikZ data files
    for data_type, data_points in plot_data.items():
        filename = f'{tikz_dir}/{data_type}.txt'
        with open(filename, 'w') as f:
            f.write(f'% {data_type} data for minimum entropy vs eta plot\n')
            f.write('% Columns: eta min_entropy\n')
            for eta, min_entropy in data_points:
                f.write(f'{eta:.2f}\t{min_entropy:.6f}\n')
    
    # Create TikZ template
    template_file = f'{tikz_dir}/plot_template.tex'
    with open(template_file, 'w') as f:
        f.write("""% TikZ template for minimum entropy vs eta plot
\\begin{tikzpicture}
\\begin{axis}[
    xlabel={$\\eta$ (Markov chain parameter)},
    ylabel={Minimum Entropy $H$ (bits)},
    title={Minimum Entropy vs $\\eta$},
    legend pos=north east,
    grid=both,
    grid style={line width=.1pt, draw=gray!10},
    major grid style={line width=.2pt,draw=gray!50},
    width=12cm, height=9cm
]

\\addplot[blue, line width=1.5pt, mark=o] table {alpha_002.txt};
\\addplot[red, line width=1.5pt, mark=square] table {alpha_006.txt};

\\legend{$\\alpha = 0.02$, $\\alpha = 0.06$}

\\end{axis}
\\end{tikzpicture}
""")
    
    print(f"  ✓ TikZ data exported to {tikz_dir}/")
    
    return fig

def create_optimal_radius_plot(results):
    """Create optimal radius vs eta plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    alpha_values = [0.02, 0.06]
    colors = {'0.02': 'blue', '0.06': 'red'}
    
    # Create TikZ data directory
    tikz_dir = 'plots/tikz_optimal_radius'
    os.makedirs(tikz_dir, exist_ok=True)
    
    plot_data = {}
    
    for alpha in alpha_values:
        eta_vals, min_entropies, opt_radii, opt_K = find_minimum_entropy_over_eta(results, alpha)
        
        ax.plot(eta_vals, opt_radii, 
                color=colors[str(alpha)], marker='s', linewidth=2.5, markersize=6,
                label=f'α = {alpha}')
        
        # Store data for TikZ
        plot_data[f'alpha_{str(alpha).replace(".", "")}'] = list(zip(eta_vals, opt_radii))
    
    ax.set_xlabel('η (Markov chain parameter)', fontsize=14)
    ax.set_ylabel('Optimal Radius R', fontsize=14)
    ax.set_title('Optimal Radius vs η', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig('plots/optimal_radius_vs_eta.png', dpi=300, bbox_inches='tight')
    plt.savefig('plots/optimal_radius_vs_eta.pdf', bbox_inches='tight')
    
    # Export TikZ data files
    for data_type, data_points in plot_data.items():
        filename = f'{tikz_dir}/{data_type}.txt'
        with open(filename, 'w') as f:
            f.write(f'% {data_type} data for optimal radius vs eta plot\n')
            f.write('% Columns: eta optimal_radius\n')
            for eta, opt_radius in data_points:
                f.write(f'{eta:.2f}\t{opt_radius:.2f}\n')
    
    # Create TikZ template
    template_file = f'{tikz_dir}/plot_template.tex'
    with open(template_file, 'w') as f:
        f.write("""% TikZ template for optimal radius vs eta plot
\\begin{tikzpicture}
\\begin{axis}[
    xlabel={$\\eta$ (Markov chain parameter)},
    ylabel={Optimal Radius $R$},
    title={Optimal Radius vs $\\eta$},
    legend pos=north east,
    grid=both,
    grid style={line width=.1pt, draw=gray!10},
    major grid style={line width=.2pt,draw=gray!50},
    width=12cm, height=9cm
]

\\addplot[blue, line width=1.5pt, mark=o] table {alpha_002.txt};
\\addplot[red, line width=1.5pt, mark=square] table {alpha_006.txt};

\\legend{$\\alpha = 0.02$, $\\alpha = 0.06$}

\\end{axis}
\\end{tikzpicture}
""")
    
    print(f"  ✓ TikZ data exported to {tikz_dir}/")
    
    return fig

def main():
    """Main function to create all plots."""
    print("Loading forgetful binary spawc results...")
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Load results
    results = load_results()
    print(f"Loaded results for alpha values: {results['alpha_values']}")
    
    # Create entropy vs radius plot (α=0.02, η=1,5, starting from K=2)
    print("\n1. Creating entropy vs radius plot (α=0.02, η=1,5)...")
    fig1 = create_entropy_vs_radius_plots(results)
    print("   ✓ Entropy vs radius plot saved to plots/entropy_vs_radius.png")
    
    # Create minimum entropy plot
    print("\n2. Creating minimum entropy vs eta plot...")
    fig2 = create_minimum_entropy_plot(results)
    print("   ✓ Minimum entropy plot saved to plots/minimum_entropy_vs_eta.png")
    
    # Create optimal radius plot
    print("\n3. Creating optimal radius vs eta plot...")
    fig3 = create_optimal_radius_plot(results)
    print("   ✓ Optimal radius plot saved to plots/optimal_radius_vs_eta.png")
    
    # Print numerical analysis
    print("\n" + "="*80)
    print("NUMERICAL ANALYSIS")
    print("="*80)
    
    alpha_values = [0.02, 0.06]
    for alpha in alpha_values:
        eta_vals, min_entropies, opt_radii, opt_K = find_minimum_entropy_over_eta(results, alpha)
        
        print(f"\nα = {alpha}:")
        
        # Find global minimum across all eta
        global_min_idx = np.argmin(min_entropies)
        global_min_eta = eta_vals[global_min_idx]
        global_min_entropy = min_entropies[global_min_idx]
        global_opt_radius = opt_radii[global_min_idx]
        global_opt_K = opt_K[global_min_idx]
        
        print(f"  Global minimum entropy: {global_min_entropy:.6f} bits")
        print(f"  Achieved at η = {global_min_eta}")
        print(f"  Optimal radius: R = {global_opt_radius}")
        print(f"  Optimal K: {global_opt_K}")
        
        # Show some specific values
        for eta_target in [1.0, 5.0, 10.0]:
            if eta_target in eta_vals:
                idx = eta_vals.index(eta_target)
                print(f"  At η={eta_target}: H_min={min_entropies[idx]:.6f}, R_opt={opt_radii[idx]}")
    
    # Don't show plots to avoid GUI issues
    plt.close('all')
    
    print(f"\n🎉 All plots created successfully!")
    print(f"📁 Check the 'plots/' directory for:")
    print(f"   - entropy_vs_radius.png/pdf + TikZ data")
    print(f"   - minimum_entropy_vs_eta.png/pdf + TikZ data") 
    print(f"   - optimal_radius_vs_eta.png/pdf + TikZ data")
    print(f"\n📊 TikZ data directories:")
    print(f"   - plots/tikz_entropy_vs_radius/")
    print(f"   - plots/tikz_minimum_entropy/")
    print(f"   - plots/tikz_optimal_radius/")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()