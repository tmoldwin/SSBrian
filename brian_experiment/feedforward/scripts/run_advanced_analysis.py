"""
Simple script to run advanced post-hoc analysis on saved simulation data.

Usage:
    python brian_experiment/run_advanced_analysis.py

This script will automatically find available simulation data and run comprehensive analysis.
"""

import os
import sys

# Add the brian_experiment directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_posthoc_analysis import AdvancedNetworkAnalyzer

def run_analysis_on_folder(folder_path):
    """Run advanced analysis on a specific folder."""
    print(f"\n🎯 Analyzing: {folder_path}")
    print("-" * 60)
    
    try:
        # Initialize analyzer
        analyzer = AdvancedNetworkAnalyzer(folder_path)
        
        # Run individual analyses
        print("\n🔬 Running Selectivity-Firing Rate Analysis...")
        
        # Analyze all populations
        all_results = analyzer.analyze_selectivity_firing_rate_correlation(population='all')
        exc_results = analyzer.analyze_selectivity_firing_rate_correlation(population='excitatory')
        inh_results = analyzer.analyze_selectivity_firing_rate_correlation(population='inhibitory')
        
        # Population evolution analysis
        print("\n🔬 Running Population Evolution Analysis...")
        evolution_results = analyzer.analyze_population_dynamics_evolution()
        
        # Clustering analysis
        print("\n🔬 Running Stimulus Response Clustering...")
        clustering_results = analyzer.analyze_stimulus_response_clustering()
        
        print(f"\n✅ Analysis completed for {folder_path}")
        
        return {
            'all_populations': all_results,
            'excitatory': exc_results, 
            'inhibitory': inh_results,
            'evolution': evolution_results,
            'clustering': clustering_results
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {folder_path}: {e}")
        return None

def main():
    """Main function to run analysis on available data."""
    print("🚀 Advanced Neural Network Analysis Runner")
    print("=" * 50)
    
    # Find available simulation data folders (try new structure first)
    base_paths = ["Runs", "Figures/DI_Sims", "brian_experiment/Figures/DI_Sims"]
    base_path = None
    
    for path in base_paths:
        if os.path.exists(path):
            base_path = path
            break
    
    if not base_path:
        print(f"❌ Cannot find simulation data directory in any of: {base_paths}")
        return
    
    # Look for folders with simulation_data.pkl (check both new and old structures)
    valid_folders = []
    for item in os.listdir(base_path):
        folder_path = os.path.join(base_path, item)
        if os.path.isdir(folder_path):
            # Check both new structure (Data subfolder) and old structure (direct)
            pkl_file_new = os.path.join(folder_path, "Data", "simulation_data.pkl")
            pkl_file_old = os.path.join(folder_path, "simulation_data.pkl")
            if os.path.exists(pkl_file_new) or os.path.exists(pkl_file_old):
                valid_folders.append(folder_path)
    
    if not valid_folders:
        print(f"❌ No simulation data found in {base_path}")
        print("Looking for folders containing 'simulation_data.pkl'")
        return
    
    print(f"📁 Found {len(valid_folders)} datasets with saved simulation data:")
    for i, folder in enumerate(valid_folders, 1):
        folder_name = os.path.basename(folder)
        print(f"   {i}. {folder_name}")
    
    # Analyze each folder
    all_results = {}
    for folder in valid_folders:
        folder_name = os.path.basename(folder)
        results = run_analysis_on_folder(folder)
        if results:
            all_results[folder_name] = results
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)
    
    for folder_name, results in all_results.items():
        print(f"\n📁 {folder_name}:")
        
        if 'all_populations' in results and results['all_populations']:
            all_pop = results['all_populations']
            exc_pop = results['excitatory'] 
            inh_pop = results['inhibitory']
            
            print(f"   🔗 Selectivity-Firing Rate Correlations:")
            print(f"      All neurons: r = {all_pop['pearson_correlation']:.3f}")
            print(f"      Excitatory:  r = {exc_pop['pearson_correlation']:.3f}")
            print(f"      Inhibitory:  r = {inh_pop['pearson_correlation']:.3f}")
            
            print(f"   📈 Mean Selectivity:")
            print(f"      Excitatory: {exc_pop['selectivity_stats']['mean']:.3f}")
            print(f"      Inhibitory: {inh_pop['selectivity_stats']['mean']:.3f}")
    
    print(f"\n✅ Advanced analysis completed!")
    print(f"📊 Results and plots saved in respective data folders")

if __name__ == "__main__":
    main()




