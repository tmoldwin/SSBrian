"""
Run ensemble analysis only from saved simulation data.
This allows regenerating ensemble analysis without rerunning the full simulation.
"""

import os
import sys
import pickle

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ensemble_analysis

def main():
    if len(sys.argv) < 2:
        # Default to the most recent 1000 neuron run
        run_folder = "Runs/direct_input_REC_stim100_ep2_n1000"
        if not os.path.exists(run_folder):
            print("Usage: python run_ensemble_only.py <run_folder>")
            print("\nExample:")
            print("  python run_ensemble_only.py Runs/direct_input_REC_stim100_ep2_n1000")
            sys.exit(1)
    else:
        run_folder = sys.argv[1]
    
    if not os.path.exists(run_folder):
        print(f"Error: Run folder not found: {run_folder}")
        sys.exit(1)
    
    data_file = os.path.join(run_folder, "Data", "simulation_data.pkl")
    if not os.path.exists(data_file):
        print(f"Error: No data file found at {data_file}")
        sys.exit(1)
    
    print(f"Loading simulation data from: {data_file}")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    config = data.get('config', {})
    
    print(f"\nRunning ensemble analysis for: {run_folder}")
    print(f"Network size: {config.get('N_rec_E', '?')}E + {config.get('N_rec_I', '?')}I neurons")
    print("="*60)
    
    stats = ensemble_analysis.generate_ensemble_analysis(run_folder, data, config)
    
    print("\n" + "="*60)
    print("Ensemble analysis complete!")
    print(f"Results saved to: {os.path.join(run_folder, 'Figures', 'ensemble_analysis')}")
    print("="*60)

if __name__ == "__main__":
    main()
