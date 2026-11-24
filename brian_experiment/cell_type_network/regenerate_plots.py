"""
Regenerate all plots from saved simulation data.

This script loads simulation data from pickle files and regenerates all figures
without re-running the simulation.

Usage:
    python regenerate_plots.py --config <config_name>
    python regenerate_plots.py <run_folder>
"""

import argparse
import pickle
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from brian_experiment.cell_type_network.configs import config_manager as cfg_manager
from brian_experiment.shared.analysis import temporal_evolution
from brian_experiment.shared.plotting.generate_plots import generate_all_plots
from brian_experiment.cell_type_network.visualize_network import draw_network_diagram
from brian_experiment.cell_type_network.plot_exemplar_by_population import plot_exemplar_neurons_by_population
from brian_experiment.cell_type_network.plot_smooth_firing_rates import plot_smooth_firing_rates_grid, plot_population_overlaid_rates
from brian_experiment.cell_type_network import network as ctn
from brian_experiment.shared.core import spike_analysis as sa
from brian2 import ms, second
import numpy as np
import matplotlib.pyplot as plt


def regenerate_all_plots(run_folder):
    """
    Regenerate all plots from saved simulation data.
    
    Parameters
    ----------
    run_folder : str
        Path to run folder containing Data/simulation_data.pkl
    """
    data_path = os.path.join(run_folder, 'Data', 'simulation_data.pkl')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Loading simulation data from: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    config = data.get('config', {})
    population_data = data.get('population_data', {})
    spike_data = data.get('spike_data', {})
    stims_by_time = data.get('stims_by_time', [])
    T = data.get('T', 0)
    
    figures_folder = os.path.join(run_folder, 'Figures')
    os.makedirs(figures_folder, exist_ok=True)
    
    print("\n" + "="*60)
    print("REGENERATING PLOTS")
    print("="*60)
    
    # Get ordered population names
    ordered_pop_names = []
    for name in sorted(population_data.keys()):
        if name != 'Layer0':
            ordered_pop_names.append(name)
    
    # 1. Generate temporal evolution plots for each population
    for pop_name in ordered_pop_names:
        if pop_name not in population_data:
            continue
        
        print(f"\n   Generating selectivity/sparsity plots for {pop_name}...")
        try:
            pop_run_folder = os.path.join(run_folder, f"analysis_{pop_name}")
            os.makedirs(pop_run_folder, exist_ok=True)
            
            pop_data = population_data[pop_name]
            pop_config = config.copy()
            pop_config['N_rec_E'] = pop_data['rate_matrix'].shape[0]
            pop_config['N_rec_I'] = 0
            
            plot_data = {
                'individual_rate_matrices': pop_data.get('individual_rate_matrices', []),
                'individual_sparsity': pop_data.get('individual_sparsity', []),
                'individual_selectivity': pop_data.get('individual_selectivity', []),
                'rate_matrices_arr': pop_data.get('rate_matrices_arr', []),
                'sparsity_vec_arr': pop_data.get('sparsity_vec_arr', []),
                'selectivity_vec_arr': pop_data.get('selectivity_vec_arr', []),
                'neuron_order': None,
                'weight_poisson': None,
                'weight_rec': None,
                'stim_info': data.get('stim_info', {}),
                'T': T,
                'config': pop_config
            }
            
            temporal_evolution.generate_temporal_evolution_plots(
                run_folder=pop_run_folder,
                data=plot_data,
                config=pop_config,
                network=None
            )
            
            # Move plots to main Figures folder
            pop_figures_src = os.path.join(pop_run_folder, "Figures", "temporal_evolution")
            if os.path.exists(pop_figures_src):
                pop_figures_dst = os.path.join(figures_folder, f"temporal_evolution_{pop_name}")
                if os.path.exists(pop_figures_dst):
                    import shutil
                    shutil.rmtree(pop_figures_dst)
                shutil.move(pop_figures_src, pop_figures_dst)
                try:
                    os.rmdir(os.path.join(pop_run_folder, "Figures"))
                    os.rmdir(pop_run_folder)
                except:
                    pass
            
            print(f"      [OK] Selectivity/sparsity plots saved for {pop_name}")
        except Exception as e:
            print(f"      [ERROR] Failed to generate selectivity plots for {pop_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 2. Network architecture diagram
    print("\n   Generating network architecture diagram...")
    try:
        network_diagram_path = os.path.join(figures_folder, 'network_architecture.png')
        draw_network_diagram(config, network_diagram_path)
    except Exception as e:
        print(f"      [ERROR] generating network diagram: {e}")
    
    # 3. Exemplar neurons by population
    print("   Generating exemplar neurons by population...")
    try:
        plot_exemplar_neurons_by_population(run_folder, n_neurons_per_pop=8, seed=42)
    except Exception as e:
        print(f"      [ERROR] generating exemplar neurons plot: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Smooth firing rates by population (exemplar neurons)
    print("   Generating smooth firing rate plots (exemplar neurons)...")
    try:
        plot_smooth_firing_rates_grid(run_folder, n_neurons_per_pop=5, seed=42, window_ms=20.0)
    except Exception as e:
        print(f"      [ERROR] generating smooth firing rate plots: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Smooth firing rates overlaid (all neurons)
    print("   Generating overlaid population smooth firing rate plots...")
    try:
        plot_population_overlaid_rates(run_folder, window_ms=20.0, max_neurons_per_panel=50)
    except Exception as e:
        print(f"      [ERROR] generating overlaid smooth firing rate plots: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Population statistics summary
    print("   Generating population statistics summary...")
    try:
        from scipy import stats
        
        fig_stats = plt.figure(figsize=(14, 10))
        gs = fig_stats.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        pop_colors = {
            'L1_E': '#E74C3C',
            'L1_I': '#3498DB',  
            'L2_E': '#E67E22',
            'L2_I': '#9B59B6',
        }
        
        pop_rates = {}
        pop_selectivity = {}
        
        for name in ordered_pop_names:
            if name in population_data:
                pop_data = population_data[name]
                if 'individual_rate_matrices' in pop_data and len(pop_data['individual_rate_matrices']) > 0:
                    rate_matrices_3d = np.array(pop_data['individual_rate_matrices'])
                    avg_rate_matrix = np.mean(rate_matrices_3d, axis=0)
                else:
                    avg_rate_matrix = pop_data['rate_matrix']
                
                selectivity = pop_data['selectivity_vec']
                neuron_rates = np.mean(avg_rate_matrix, axis=1)
                neuron_rates = neuron_rates[np.isfinite(neuron_rates)]
                selectivity = selectivity[np.isfinite(selectivity)]
                
                if len(neuron_rates) > 1:
                    pop_rates[name] = neuron_rates
                if len(selectivity) > 1:
                    pop_selectivity[name] = selectivity
        
        # Panel 1: Firing Rate KDE
        ax1 = fig_stats.add_subplot(gs[0, 0])
        if pop_rates:
            max_rate = max(r.max() for r in pop_rates.values())
            x_range = np.linspace(0, max_rate * 1.1, 200)
            for name, rates in pop_rates.items():
                color = pop_colors.get(name, '#95A5A6')
                kde = stats.gaussian_kde(rates)
                ax1.fill_between(x_range, kde(x_range), alpha=0.4, color=color, label=f'{name} (n={len(rates)})')
                ax1.plot(x_range, kde(x_range), color=color, linewidth=2)
        ax1.set_xlabel('Firing Rate (Hz)', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('Firing Rate Distribution by Population', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.set_xlim(left=0)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Selectivity KDE
        ax2 = fig_stats.add_subplot(gs[0, 1])
        x_sel = np.linspace(0, 1.1, 200)
        for name, sel in pop_selectivity.items():
            color = pop_colors.get(name, '#95A5A6')
            kde = stats.gaussian_kde(sel)
            ax2.fill_between(x_sel, kde(x_sel), alpha=0.4, color=color, label=f'{name} (n={len(sel)})')
            ax2.plot(x_sel, kde(x_sel), color=color, linewidth=2)
        ax2.set_xlabel('Selectivity Index (SLI)', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.set_title('Selectivity Distribution by Population', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.set_xlim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3 & 4: Bar charts
        colors = [pop_colors.get(n, '#95A5A6') for n in ordered_pop_names]
        rates_mean = [np.mean(pop_rates[n]) for n in ordered_pop_names if n in pop_rates]
        sel_mean = [np.mean(pop_selectivity[n]) for n in ordered_pop_names if n in pop_selectivity]
        pop_names_plot = [n for n in ordered_pop_names if n in pop_rates]
        
        ax3 = fig_stats.add_subplot(gs[1, 0])
        bars = ax3.bar(range(len(pop_names_plot)), rates_mean, color=colors, alpha=0.8, edgecolor='black')
        ax3.set_xticks(range(len(pop_names_plot)))
        ax3.set_xticklabels(pop_names_plot, fontsize=10)
        ax3.set_ylabel('Mean Firing Rate (Hz)', fontsize=11)
        ax3.set_title('Mean Firing Rate by Population', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        for bar, r in zip(bars, rates_mean): 
            ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{r:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax4 = fig_stats.add_subplot(gs[1, 1])
        bars = ax4.bar(range(len(pop_names_plot)), sel_mean, color=colors, alpha=0.8, edgecolor='black')
        ax4.set_xticks(range(len(pop_names_plot)))
        ax4.set_xticklabels(pop_names_plot, fontsize=10)
        ax4.set_ylabel('Mean Selectivity (SLI)', fontsize=11)
        ax4.set_title('Mean Selectivity by Population', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 1.0)
        for bar, s in zip(bars, sel_mean): 
            ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{s:.2f}', ha='center', va='bottom', fontsize=9)
        
        fig_stats.suptitle('Population Statistics Summary', fontsize=14, fontweight='bold', y=0.98)
        fig_stats.tight_layout(rect=[0, 0, 1, 0.96])
        fig_stats.savefig(os.path.join(figures_folder, 'population_statistics.png'), dpi=150, bbox_inches='tight')
        plt.close(fig_stats)
        print(f"      Saved: population_statistics.png")
    except Exception as e:
        print(f"      [ERROR] generating population statistics: {e}")
        import traceback
        traceback.print_exc()
    
    # Note: Other plots (raster, weight matrix, voltage traces) require the network object
    # which isn't easily reconstructable from pickle. Those are generated during simulation.
    
    print(f"\n>> Plot regeneration complete!")
    print(f"   Plots saved to: {figures_folder}")


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate all plots from saved simulation data'
    )
    parser.add_argument(
        'run_folder',
        nargs='?',
        type=str,
        help='Path to run folder containing Data/simulation_data.pkl'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Config name (e.g., test_two_layers). Will look in Runs/single_runs/<config>/latest'
    )
    
    args = parser.parse_args()
    
    # Resolve run folder path
    if args.config:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        run_folder = os.path.join(script_dir, 'Runs', 'single_runs', args.config, 'latest')
    elif args.run_folder:
        run_folder = args.run_folder
    else:
        parser.error("Must provide either run_folder or --config")
    
    if not os.path.exists(run_folder):
        print(f"Error: Run folder not found: {run_folder}")
        sys.exit(1)
    
    try:
        regenerate_all_plots(run_folder)
    except Exception as e:
        print(f"Error regenerating plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

