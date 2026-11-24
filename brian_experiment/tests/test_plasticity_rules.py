"""
Unit tests for plasticity rules with visualization.

This script tests each plasticity rule and creates a multi-panel figure showing:
1. How weight change varies with temporal difference (delta_t)
2. The direction of plasticity (potentiation vs depression)
3. Time constants and decay behavior

Each plasticity rule has its own visualize() method that returns a panel (axes).
"""

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import os
from plasticity_kernels import STDPKernel, HebbianKernel, AntiHebbianKernel, CustomKernel

# Set backend
prefs.codegen.target = "numpy"

# Output directory for all plasticity plots
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plasticity")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_all_plasticity_panels():
    """Create a multi-panel figure showing all plasticity rules."""
    print("="*60)
    print("Creating Multi-Panel Plasticity Visualization")
    print("="*60)
    
    # Create all kernels with tunable width parameters
    # STDP: tunable widths for LTP and LTD bumps
    stdp = STDPKernel(
        eta_pre=0.5,       # LTP scale
        eta_post=0.525,    # LTD scale (slightly larger for balance)
        tau_pre=20*ms,     # LTP width (time constant) - TUNABLE
        tau_post=20*ms,    # LTD width (time constant) - TUNABLE
        weight_bounds=None
    )
    
    # Hebbian: tunable widths for Mexican hat function
    hebbian = HebbianKernel(
        A_hebb=0.01,         # Potentiation amplitude
        tau_hebb=20*ms,      # Narrow bump width (time constant) - TUNABLE
        tau_hebb_wide=40*ms, # Wide depression width (time constant) - TUNABLE
        learning_rate=1.0
    )
    
    # Anti-Hebbian: tunable width for symmetric inverted bump
    anti_hebbian = AntiHebbianKernel(
        A_anti=0.01,     # Depression amplitude
        tau_anti=20*ms,  # Bump width (time constant) - TUNABLE
        learning_rate=1.0
    )
    
    # Define a custom kernel function (e.g., symmetric STDP)
    def symmetric_stdp(delta_t):
        """Symmetric STDP: same magnitude for LTP and LTD."""
        # Handle Brian2 Quantity arrays
        if isinstance(delta_t, np.ndarray) or hasattr(delta_t, '__len__'):
            # Convert to plain numpy array (in milliseconds)
            if hasattr(delta_t, 'magnitude'):
                delta_t_vals = np.asarray(delta_t / ms, dtype=float)
            else:
                delta_t_vals = np.array([float(dt) if hasattr(dt, '__float__') else dt for dt in delta_t], dtype=float)
            
            result = np.zeros_like(delta_t_vals, dtype=float)
            mask_positive = delta_t_vals > 0
            mask_negative = delta_t_vals < 0
            tau = 20.0  # ms
            A = 0.01
            result[mask_positive] = A * np.exp(-delta_t_vals[mask_positive] / tau)
            result[mask_negative] = -A * np.exp(delta_t_vals[mask_negative] / tau)
            return result
        else:
            # Scalar case
            if hasattr(delta_t, 'magnitude'):
                delta_t_val = float(delta_t / ms)
            else:
                delta_t_val = float(delta_t) if hasattr(delta_t, '__float__') else delta_t
            
            tau = 20.0  # ms
            A = 0.01
            if delta_t_val > 0:
                return A * np.exp(-delta_t_val / tau)
            elif delta_t_val < 0:
                return -A * np.exp(delta_t_val / tau)
            else:
                return 0.0
    
    custom = CustomKernel(
        kernel_function=symmetric_stdp,
        learning_rate=1.0
    )
    
    # Create figure with subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Panel 1: STDP (up bump and down bump)
    print("\nPlotting STDP kernel...")
    stdp.visualize(delta_t_range=(-100, 100), n_points=1000, ax=axes[0])
    
    # Panel 2: Hebbian (symmetric bump)
    print("Plotting Hebbian kernel...")
    hebbian.visualize(delta_t_range=(-100, 100), n_points=1000, ax=axes[1])
    
    # Panel 3: Anti-Hebbian (symmetric inverted bump)
    print("Plotting Anti-Hebbian kernel...")
    anti_hebbian.visualize(delta_t_range=(-100, 100), n_points=1000, ax=axes[2])
    
    # Panel 4: Custom (symmetric STDP)
    print("Plotting Custom kernel...")
    # Use base class visualize for custom kernel
    custom.visualize(delta_t_range=(-100, 100), n_points=1000, ax=axes[3], 
                     label='Symmetric STDP (Custom)', color='magenta')
    axes[3].set_title('Custom Kernel: Symmetric STDP', fontsize=14, fontweight='bold')
    axes[3].legend()
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'test_all_plasticity_panels.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {out_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Plasticity Rule Statistics")
    print("="*60)
    print(f"\nSTDP:")
    print(f"  eta_pre (LTP): {stdp.eta_pre}")
    print(f"  eta_post (LTD): {stdp.eta_post}")
    print(f"  tau_pre: {stdp.tau_pre/ms} ms")
    print(f"  tau_post: {stdp.tau_post/ms} ms")
    
    print(f"\nHebbian (Mexican Hat):")
    print(f"  A_hebb: {hebbian.A_hebb}")
    print(f"  tau_hebb (narrow): {hebbian.tau_hebb/ms} ms")
    print(f"  tau_hebb_wide: {hebbian.tau_hebb_wide/ms} ms")
    print(f"  Note: In-sync strengthen, out-of-sync weaken")
    
    print(f"\nAnti-Hebbian:")
    print(f"  A_anti: {anti_hebbian.A_anti}")
    print(f"  tau_anti: {anti_hebbian.tau_anti/ms} ms")
    print(f"  Note: Symmetric in time")
    
    return fig, axes


def compare_all_kernels():
    """Compare all plasticity kernels on a single panel."""
    print("\n" + "="*60)
    print("Comparing All Plasticity Kernels")
    print("="*60)
    
    # Create all kernels
    stdp = STDPKernel(eta_pre=0.5, eta_post=0.525, tau_pre=20*ms, tau_post=20*ms)
    hebbian = HebbianKernel(A_hebb=0.01, tau_hebb=20*ms, tau_hebb_wide=40*ms)
    anti_hebbian = AntiHebbianKernel(A_anti=0.01, tau_anti=20*ms)
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Use each kernel's visualize method
    stdp.visualize(delta_t_range=(-100, 100), n_points=1000, ax=ax, color='blue', label='STDP')
    hebbian.visualize(delta_t_range=(-100, 100), n_points=1000, ax=ax, color='green', label='Hebbian')
    anti_hebbian.visualize(delta_t_range=(-100, 100), n_points=1000, ax=ax, color='red', label='Anti-Hebbian')
    
    # Update title
    ax.set_title('Comparison of Plasticity Kernels', fontsize=14, fontweight='bold')
    
    # Add region labels
    ax.text(30, 0.008, 'Pre → Post', fontsize=10, ha='left',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(-30, -0.008, 'Post → Pre', fontsize=10, ha='right',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'test_all_kernels_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {out_path}")
    
    return fig, ax


def test_time_constants():
    """Test how different time constants affect plasticity."""
    print("\n" + "="*60)
    print("Testing Effect of Time Constants")
    print("="*60)
    
    # Create STDP kernels with different time constants
    tau_values = [10*ms, 20*ms, 40*ms]
    colors = ['blue', 'green', 'red']
    labels = ['τ = 10 ms', 'τ = 20 ms', 'τ = 40 ms']
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot LTP side (positive delta_t)
    for tau, color, label in zip(tau_values, colors, labels):
        stdp = STDPKernel(eta_pre=0.5, eta_post=0.525, tau_pre=tau, tau_post=tau)
        stdp.visualize(delta_t_range=(0, 100), n_points=1000, ax=ax1, color=color, label=label)
    
    ax1.set_title('LTP: Effect of Width (τ)', fontsize=13, fontweight='bold')
    ax1.set_xlabel(r'$\Delta t$ (ms) [t_post - t_pre]', fontsize=12)
    ax1.set_ylabel('Weight Change (LTP)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot LTD side (negative delta_t)
    for tau, color, label in zip(tau_values, colors, labels):
        stdp = STDPKernel(eta_pre=0.5, eta_post=0.525, tau_pre=tau, tau_post=tau)
        stdp.visualize(delta_t_range=(-100, 0), n_points=1000, ax=ax2, color=color, label=label)
    
    ax2.set_title('LTD: Effect of Width (τ)', fontsize=13, fontweight='bold')
    ax2.set_xlabel(r'$\Delta t$ (ms) [t_post - t_pre]', fontsize=12)
    ax2.set_ylabel('Weight Change (LTD)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'test_time_constants.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {out_path}")
    
    return fig, (ax1, ax2)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PLASTICITY RULES UNIT TESTS")
    print("="*60)
    print("\nThis script tests all plasticity rules and creates visualization plots.")
    print("Each plasticity rule has its own visualize() method that returns a panel.")
    print("The test script creates multi-panel figures to compare rules.")
    print("\n")
    
    # Create main multi-panel figure
    create_all_plasticity_panels()
    
    # Create comparison plot
    compare_all_kernels()
    
    # Test time constants
    test_time_constants()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
    print("\nGenerated plots directory:")
    print(f"  - {OUTPUT_DIR}")
    print("\n")
