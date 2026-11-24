cimport numpy as np
import matplotlib.pyplot as plt
from brian_experiment.sparsity_and_selectivity import selectivity_index, sparsity_index
from Scaling_experiments import window_responses, consistency_ratio

def generate_example_responses(n_neurons=100, n_stimuli=10, n_trials=20, 
                             sparsity_levels=[0.1, 0.5, 0.9],
                             selectivity_levels=[0.2, 0.6, 0.9],
                             noise_levels=[0.1, 0.3, 0.5]):
    """
    Generate example neural responses with different sparsity/selectivity/noise levels
    
    Args:
        n_neurons: number of neurons
        n_stimuli: number of different stimuli
        n_trials: number of trials per stimulus
        sparsity_levels: fraction of stimuli each neuron responds to
        selectivity_levels: how different the responses are to different stimuli
        noise_levels: trial-to-trial variability
        
    Returns:
        responses: dict of response arrays (n_neurons, n_stimuli, n_trials)
        with different parameter combinations
    """
    responses = {}
    
    for sparsity in sparsity_levels:
        for selectivity in selectivity_levels:
            for noise in noise_levels:
                # Generate base responses
                responses_clean = np.zeros((n_neurons, n_stimuli))
                
                # For each neuron
                for n in range(n_neurons):
                    # Select which stimuli this neuron responds to
                    n_active = int(n_stimuli * (1-sparsity))
                    active_stimuli = np.random.choice(n_stimuli, n_active, replace=False)
                    
                    # Generate different response levels based on selectivity
                    for s in active_stimuli:
                        if selectivity > 0:
                            responses_clean[n, s] = np.random.uniform(0, selectivity)
                        else:
                            responses_clean[n, s] = 1.0
                
                # Add trial-to-trial variability
                responses_noisy = np.zeros((n_neurons, n_stimuli, n_trials))
                for t in range(n_trials):
                    responses_noisy[:, :, t] = responses_clean + np.random.normal(0, noise, responses_clean.shape)
                
                key = f"sparsity_{sparsity:.1f}_selectivity_{selectivity:.1f}_noise_{noise:.1f}"
                responses[key] = responses_noisy
    
    return responses

def plot_metric_relationships(responses):
    """Plot relationships between sparsity, selectivity, and consistency"""
    n_conditions = len(responses)
    fig, axes = plt.subplots(3, 1, figsize=(15, 20))
    
    # Collect metrics for all conditions
    all_sparsity = []
    all_selectivity = []
    all_consistency = []
    labels = []
    
    for key, resp in responses.items():
        # Calculate metrics
        sparsity = np.mean([sparsity_index(i, resp.mean(axis=2)) 
                           for i in range(resp.shape[1])])
        selectivity = np.mean([selectivity_index(i, resp.mean(axis=2)) 
                             for i in range(resp.shape[0])])
        consistency = np.mean(consistency_ratio(resp))
        
        all_sparsity.append(sparsity)
        all_selectivity.append(selectivity)
        all_consistency.append(consistency)
        labels.append(key)
    
    # Plot relationships
    axes[0].scatter(all_sparsity, all_selectivity)
    axes[0].set_xlabel('Sparsity')
    axes[0].set_ylabel('Selectivity')
    axes[0].set_title('Sparsity vs Selectivity')
    
    axes[1].scatter(all_sparsity, all_consistency)
    axes[1].set_xlabel('Sparsity')
    axes[1].set_ylabel('Consistency')
    axes[1].set_title('Sparsity vs Consistency')
    
    axes[2].scatter(all_selectivity, all_consistency)
    axes[2].set_xlabel('Selectivity')
    axes[2].set_ylabel('Consistency')
    axes[2].set_title('Selectivity vs Consistency')
    
    plt.tight_layout()
    plt.show()

# Generate and plot example data
responses = generate_example_responses()
plot_metric_relationships(responses) 