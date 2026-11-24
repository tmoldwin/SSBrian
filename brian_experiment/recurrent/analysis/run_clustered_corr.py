"""Run clustered correlation matrix analysis on a specific run."""

import sys
import os
import pickle

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import correlation_analysis

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to 200 neuron run
        run_folder = "Runs/direct_input_REC_stim100_ep2_n200"
    else:
        run_folder = sys.argv[1]
    
    data_file = os.path.join(run_folder, "Data", "simulation_data.pkl")
    if not os.path.exists(data_file):
        print(f"Error: No data file found at {data_file}")
        sys.exit(1)
    
    print(f"Loading data from: {data_file}")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    config = data.get('config', {})
    
    print(f"Running clustered correlation matrix analysis...")
    stats = correlation_analysis.plot_clustered_correlation_matrix(run_folder, data, config)
    
    print(f"\nAnalysis complete!")

