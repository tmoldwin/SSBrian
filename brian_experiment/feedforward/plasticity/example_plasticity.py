"""
Example script demonstrating how to use the plasticity system.

This script shows how to configure different plasticity rules for different
connection types in the network.
"""

from brian2 import *
from plasticity_kernels import STDPKernel, HebbianKernel, AntiHebbianKernel, CustomKernel
from brian_experiment.feedforward import network as din
import numpy as np

# Set backend
prefs.codegen.target = "numpy"

# Example 1: STDP on E->E connections only
print("="*60)
print("Example 1: STDP on E->E connections")
print("="*60)

# Create STDP kernel
stdp_kernel = STDPKernel(
    A_pre=0.01, 
    A_post=-0.0105, 
    tau_pre=20*ms, 
    tau_post=20*ms,
    learning_rate=1.0,
    weight_bounds=(0, 0.1*nA)  # Keep weights positive and bounded
)

# Configure plasticity: only E->E gets STDP
plasticity_config_1 = {
    'E->E': stdp_kernel,
    'E->I': None,
    'I->E': None,
    'I->I': None,
    'input': None
}

# Example 2: Different rules for different connection types
print("\n" + "="*60)
print("Example 2: Mixed plasticity rules")
print("="*60)

# Create multiple kernels
stdp_kernel_2 = STDPKernel(A_pre=0.01, A_post=-0.0105, tau_pre=20*ms, tau_post=20*ms)
hebbian_kernel = HebbianKernel(A_hebb=0.01, tau_hebb=20*ms, learning_rate=0.5)
anti_hebbian_kernel = AntiHebbianKernel(A_anti=0.01, tau_anti=20*ms, learning_rate=0.5)

# Configure plasticity: different rules for each connection type
plasticity_config_2 = {
    'E->E': stdp_kernel_2,        # STDP on E->E
    'E->I': hebbian_kernel,       # Hebbian on E->I
    'I->E': None,                 # No plasticity on I->E
    'I->I': anti_hebbian_kernel,  # Anti-Hebbian on I->I
    'input': None
}

# Example 3: Custom plasticity kernel
print("\n" + "="*60)
print("Example 3: Custom plasticity kernel")
print("="*60)

def custom_kernel_function(delta_t):
    """
    Custom plasticity rule: exponential decay with asymmetric windows.
    """
    if isinstance(delta_t, np.ndarray):
        result = np.zeros_like(delta_t)
        # Positive window (pre before post)
        mask_pos = (delta_t > 0) & (delta_t < 50*ms)
        result[mask_pos] = 0.01 * np.exp(-delta_t[mask_pos] / (10*ms))
        # Negative window (post before pre)
        mask_neg = (delta_t < 0) & (delta_t > -50*ms)
        result[mask_neg] = -0.005 * np.exp(delta_t[mask_neg] / (20*ms))
        return result
    else:
        if 0 < delta_t < 50*ms:
            return 0.01 * np.exp(-delta_t / (10*ms))
        elif -50*ms < delta_t < 0:
            return -0.005 * np.exp(delta_t / (20*ms))
        else:
            return 0.0

custom_kernel = CustomKernel(
    kernel_function=custom_kernel_function,
    learning_rate=1.0,
    weight_bounds=(0, 0.1*nA)
)

plasticity_config_3 = {
    'E->E': custom_kernel,
    'E->I': None,
    'I->E': None,
    'I->I': None,
    'input': None
}

# Example usage in network initialization:
# 
# N_rec_E = 100
# N_rec_I = 100
# N_external_input_dims = 500
# weight_rec = ...  # Your weight matrix
# input_weight_scale = 1.0
# 
# # Create network with plasticity
# network = din.DirectInputNetwork(
#     N_rec_E, N_rec_I, N_external_input_dims, 
#     weight_rec, input_weight_scale,
#     lognormal_params_in=None,
#     is_feedforward=False,
#     plasticity_config=plasticity_config_1  # or plasticity_config_2, plasticity_config_3
# )

# Example 4: Visualize plasticity kernels
print("\n" + "="*60)
print("Example 4: Visualize plasticity kernels")
print("="*60)

import matplotlib.pyplot as plt

# Create kernels
stdp_viz = STDPKernel(A_pre=0.01, A_post=-0.0105, tau_pre=20*ms, tau_post=20*ms)
hebbian_viz = HebbianKernel(A_hebb=0.01, tau_hebb=20*ms)
anti_hebbian_viz = AntiHebbianKernel(A_anti=0.01, tau_anti=20*ms)

# Visualize them
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

stdp_viz.visualize(delta_t_range=(-50, 50), ax=axes[0], label='STDP', color='blue')
hebbian_viz.visualize(delta_t_range=(-50, 50), ax=axes[1], label='Hebbian', color='green')
anti_hebbian_viz.visualize(delta_t_range=(-50, 50), ax=axes[2], label='Anti-Hebbian', color='red')

plt.tight_layout()
plt.savefig('plasticity_kernels_visualization.png', dpi=150, bbox_inches='tight')
print("Saved plasticity kernel visualization to 'plasticity_kernels_visualization.png'")

print("\n" + "="*60)
print("Plasticity configuration examples created!")
print("="*60)
print("\nTo use plasticity in your network:")
print("1. Import the plasticity kernels you need")
print("2. Create kernel instances with desired parameters")
print("3. Create a plasticity_config dictionary")
print("4. Pass plasticity_config to DirectInputNetwork.__init__()")
print("\nAvailable connection types:")
print("  - 'E->E': Excitatory to excitatory")
print("  - 'E->I': Excitatory to inhibitory")
print("  - 'I->E': Inhibitory to excitatory")
print("  - 'I->I': Inhibitory to inhibitory")
print("  - 'input->E': Feed-forward input to excitatory neurons")
print("  - 'input->I': Feed-forward input to inhibitory neurons")
print("\nTo visualize a plasticity kernel:")
print("  kernel.visualize(delta_t_range=(-50, 50), n_points=200)")

