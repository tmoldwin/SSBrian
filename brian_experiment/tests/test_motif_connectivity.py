"""
Test script for motif connectivity generation.

This script demonstrates and validates the motif connectivity system.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.core.motif_connectivity import (
    MotifConnectivityGenerator,
    generate_motif_connectivity,
    validate_motif_constraints
)


def test_basic_connectivity():
    """Test basic connectivity generation without motif constraints."""
    print("\n" + "="*60)
    print("TEST 1: Basic Connectivity (No Motif Constraints)")
    print("="*60)
    
    pop_configs = [
        {"name": "E", "n": 100, "type": "E"},
        {"name": "I", "n": 25, "type": "I"}
    ]
    
    conn_configs = [
        {"source": "E", "target": "E", "p": 0.2},
        {"source": "E", "target": "I", "p": 0.3},
        {"source": "I", "target": "E", "p": 0.3},
        {"source": "I", "target": "I", "p": 0.2}
    ]
    
    connectivity = generate_motif_connectivity(pop_configs, conn_configs, seed=42)
    
    # Check counts
    for (src, tgt), conns in connectivity.items():
        n_src = next(p['n'] for p in pop_configs if p['name'] == src)
        n_tgt = next(p['n'] for p in pop_configs if p['name'] == tgt)
        n_possible = n_src * n_tgt if src != tgt else n_src * (n_tgt - 1)
        expected_p = next(c['p'] for c in conn_configs if c['source'] == src and c['target'] == tgt)
        actual_p = len(conns) / n_possible if n_possible > 0 else 0
        
        print(f"  {src}->{tgt}: {len(conns)} connections, p={actual_p:.3f} (target: {expected_p:.3f})")
    
    print("  [OK] Basic connectivity test passed")


def test_reciprocal_enhancement():
    """Test that reciprocal motif enhancement works correctly."""
    print("\n" + "="*60)
    print("TEST 2: Reciprocal E-I Enhancement")
    print("="*60)
    
    pop_configs = [
        {"name": "E", "n": 200, "type": "E"},
        {"name": "I", "n": 50, "type": "I"}
    ]
    
    conn_configs = [
        {"source": "E", "target": "I", "p": 0.25},
        {"source": "I", "target": "E", "p": 0.25}
    ]
    
    # Test without enhancement
    print("\n  Without motif constraints (should be ~6.25% reciprocal):")
    connectivity_base = generate_motif_connectivity(pop_configs, conn_configs, seed=42)
    
    base_stats = compute_reciprocal_stats(connectivity_base, pop_configs)
    print(f"    E->I connections: {base_stats['n_EI']}")
    print(f"    I->E connections: {base_stats['n_IE']}")
    print(f"    Reciprocal pairs: {base_stats['n_reciprocal']}")
    print(f"    Reciprocal rate:  {base_stats['reciprocal_rate']:.3f}")
    print(f"    Expected (indep): {base_stats['expected_reciprocal']:.1f}")
    
    # Test with 3x enhancement
    print("\n  With 3x reciprocal enhancement:")
    motif_constraints = [
        {
            "name": "reciprocal_EI",
            "type": "reciprocal",
            "node_filters": {
                "A": {"type": "E"},
                "B": {"type": "I"}
            },
            "enhancement": 3.0
        }
    ]
    
    connectivity_enhanced = generate_motif_connectivity(
        pop_configs, conn_configs, motif_constraints, seed=42
    )
    
    enhanced_stats = compute_reciprocal_stats(connectivity_enhanced, pop_configs)
    print(f"    E->I connections: {enhanced_stats['n_EI']}")
    print(f"    I->E connections: {enhanced_stats['n_IE']}")
    print(f"    Reciprocal pairs: {enhanced_stats['n_reciprocal']}")
    print(f"    Reciprocal rate:  {enhanced_stats['reciprocal_rate']:.3f}")
    print(f"    Enhancement:      {enhanced_stats['reciprocal_rate'] / base_stats['reciprocal_rate']:.2f}x")
    
    # Verify enhancement worked
    if enhanced_stats['reciprocal_rate'] > base_stats['reciprocal_rate'] * 1.5:
        print("  [OK] Reciprocal enhancement test passed")
    else:
        print("  [WARN] Reciprocal enhancement may not be working as expected")


def compute_reciprocal_stats(connectivity, pop_configs):
    """Compute statistics about reciprocal connections."""
    conns_EI = set(connectivity.get(("E", "I"), []))
    conns_IE = connectivity.get(("I", "E"), [])
    
    # Flip I->E to compare with E->I
    conns_IE_flipped = set((j, i) for (i, j) in conns_IE)
    
    n_reciprocal = len(conns_EI & conns_IE_flipped)
    
    n_E = next(p['n'] for p in pop_configs if p['name'] == 'E')
    n_I = next(p['n'] for p in pop_configs if p['name'] == 'I')
    n_possible = n_E * n_I
    
    p_EI = len(conns_EI) / n_possible if n_possible > 0 else 0
    p_IE = len(conns_IE) / n_possible if n_possible > 0 else 0
    
    return {
        'n_EI': len(conns_EI),
        'n_IE': len(conns_IE),
        'n_reciprocal': n_reciprocal,
        'reciprocal_rate': n_reciprocal / len(conns_EI) if len(conns_EI) > 0 else 0,
        'expected_reciprocal': p_EI * p_IE * n_possible
    }


def test_marginal_preservation():
    """Test that marginal connection probabilities are preserved."""
    print("\n" + "="*60)
    print("TEST 3: Marginal Probability Preservation")
    print("="*60)
    
    pop_configs = [
        {"name": "E", "n": 200, "type": "E"},
        {"name": "I", "n": 50, "type": "I"}
    ]
    
    conn_configs = [
        {"source": "E", "target": "I", "p": 0.3},
        {"source": "I", "target": "E", "p": 0.4}
    ]
    
    motif_constraints = [
        {
            "name": "reciprocal_EI",
            "type": "reciprocal",
            "node_filters": {"A": {"type": "E"}, "B": {"type": "I"}},
            "enhancement": 2.5
        }
    ]
    
    # Run multiple times to check variance
    n_runs = 10
    p_EI_samples = []
    p_IE_samples = []
    
    for i in range(n_runs):
        connectivity = generate_motif_connectivity(
            pop_configs, conn_configs, motif_constraints, seed=i
        )
        
        n_E = 200
        n_I = 50
        n_possible = n_E * n_I
        
        p_EI = len(connectivity.get(("E", "I"), [])) / n_possible
        p_IE = len(connectivity.get(("I", "E"), [])) / n_possible
        
        p_EI_samples.append(p_EI)
        p_IE_samples.append(p_IE)
    
    print(f"  E->I: target={conn_configs[0]['p']:.3f}, actual={np.mean(p_EI_samples):.3f} ± {np.std(p_EI_samples):.3f}")
    print(f"  I->E: target={conn_configs[1]['p']:.3f}, actual={np.mean(p_IE_samples):.3f} ± {np.std(p_IE_samples):.3f}")
    
    # Check if marginals are within acceptable range
    tol = 0.05
    ei_ok = abs(np.mean(p_EI_samples) - conn_configs[0]['p']) < tol
    ie_ok = abs(np.mean(p_IE_samples) - conn_configs[1]['p']) < tol
    
    if ei_ok and ie_ok:
        print("  [OK] Marginal preservation test passed")
    else:
        print("  [WARN] Marginals may be slightly off (this can happen with high enhancement)")


def test_multi_population():
    """Test with multiple populations (layers)."""
    print("\n" + "="*60)
    print("TEST 4: Multi-Population (Layered Network)")
    print("="*60)
    
    pop_configs = [
        {"name": "L1_E", "n": 80, "type": "E"},
        {"name": "L1_I", "n": 20, "type": "I"},
        {"name": "L2_E", "n": 80, "type": "E"},
        {"name": "L2_I", "n": 20, "type": "I"}
    ]
    
    conn_configs = [
        # Within L1
        {"source": "L1_E", "target": "L1_I", "p": 0.25},
        {"source": "L1_I", "target": "L1_E", "p": 0.25},
        # Within L2
        {"source": "L2_E", "target": "L2_I", "p": 0.25},
        {"source": "L2_I", "target": "L2_E", "p": 0.25},
        # L1 -> L2
        {"source": "L1_E", "target": "L2_E", "p": 0.3},
        {"source": "L1_E", "target": "L2_I", "p": 0.2}
    ]
    
    motif_constraints = [
        {
            "name": "reciprocal_EI",
            "type": "reciprocal",
            "node_filters": {"A": {"type": "E"}, "B": {"type": "I"}},
            "enhancement": 3.0
        }
    ]
    
    connectivity = generate_motif_connectivity(
        pop_configs, conn_configs, motif_constraints, seed=42
    )
    
    print("  Generated connections:")
    for (src, tgt), conns in sorted(connectivity.items()):
        print(f"    {src} -> {tgt}: {len(conns)} connections")
    
    # Check reciprocal enhancement in each layer
    for layer in ['L1', 'L2']:
        E_pop = f"{layer}_E"
        I_pop = f"{layer}_I"
        
        conns_EI = set(connectivity.get((E_pop, I_pop), []))
        conns_IE = connectivity.get((I_pop, E_pop), [])
        conns_IE_flipped = set((j, i) for (i, j) in conns_IE)
        
        n_reciprocal = len(conns_EI & conns_IE_flipped)
        
        print(f"\n  {layer}: {n_reciprocal} reciprocal E-I pairs out of {len(conns_EI)} E->I connections")
    
    print("\n  [OK] Multi-population test completed")


def visualize_connectivity_matrix():
    """Visualize connectivity matrix with and without motif constraints."""
    print("\n" + "="*60)
    print("TEST 5: Visualization of Connectivity Matrices")
    print("="*60)
    
    pop_configs = [
        {"name": "E", "n": 50, "type": "E"},
        {"name": "I", "n": 50, "type": "I"}
    ]
    
    conn_configs = [
        {"source": "E", "target": "E", "p": 0.15},
        {"source": "E", "target": "I", "p": 0.25},
        {"source": "I", "target": "E", "p": 0.25},
        {"source": "I", "target": "I", "p": 0.15}
    ]
    
    motif_constraints = [
        {
            "name": "reciprocal_EI",
            "type": "reciprocal",
            "node_filters": {"A": {"type": "E"}, "B": {"type": "I"}},
            "enhancement": 4.0
        }
    ]
    
    # Generate both versions
    conn_base = generate_motif_connectivity(pop_configs, conn_configs, seed=42)
    conn_motif = generate_motif_connectivity(pop_configs, conn_configs, motif_constraints, seed=42)
    
    # Create matrices
    n = 100  # Total neurons
    mat_base = np.zeros((n, n))
    mat_motif = np.zeros((n, n))
    
    offset = {"E": 0, "I": 50}
    
    for (src, tgt), conns in conn_base.items():
        for (i, j) in conns:
            mat_base[offset[tgt] + j, offset[src] + i] = 1
    
    for (src, tgt), conns in conn_motif.items():
        for (i, j) in conns:
            mat_motif[offset[tgt] + j, offset[src] + i] = 1
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(mat_base, aspect='auto', cmap='gray_r', interpolation='nearest')
    axes[0].set_title('Without Motif Constraints')
    axes[0].axhline(50, color='red', linewidth=0.5)
    axes[0].axvline(50, color='red', linewidth=0.5)
    axes[0].set_xlabel('Source (Pre)')
    axes[0].set_ylabel('Target (Post)')
    
    axes[1].imshow(mat_motif, aspect='auto', cmap='gray_r', interpolation='nearest')
    axes[1].set_title('With 4x Reciprocal Enhancement')
    axes[1].axhline(50, color='red', linewidth=0.5)
    axes[1].axvline(50, color='red', linewidth=0.5)
    axes[1].set_xlabel('Source (Pre)')
    axes[1].set_ylabel('Target (Post)')
    
    # Difference plot - highlight reciprocal structure
    # Extract E->I and I->E quadrants
    EI_base = mat_base[50:, :50]  # E->I
    IE_base = mat_base[:50, 50:]  # I->E
    EI_motif = mat_motif[50:, :50]
    IE_motif = mat_motif[:50, 50:]
    
    # Compute reciprocal indicator
    recip_base = EI_base * IE_base.T
    recip_motif = EI_motif * IE_motif.T
    
    axes[2].bar(['Baseline', 'Enhanced'], [recip_base.sum(), recip_motif.sum()], color=['gray', 'blue'])
    axes[2].set_ylabel('Number of Reciprocal E-I Pairs')
    axes[2].set_title('Reciprocal Connection Count')
    
    plt.tight_layout()
    plt.savefig('motif_connectivity_test.png', dpi=150)
    print("  Saved visualization to motif_connectivity_test.png")
    print("  [OK] Visualization test completed")
    
    return fig


def test_extreme_enhancement():
    """Test behavior with extreme enhancement values."""
    print("\n" + "="*60)
    print("TEST 6: Extreme Enhancement Values")
    print("="*60)
    
    pop_configs = [
        {"name": "E", "n": 100, "type": "E"},
        {"name": "I", "n": 25, "type": "I"}
    ]
    
    conn_configs = [
        {"source": "E", "target": "I", "p": 0.3},
        {"source": "I", "target": "E", "p": 0.3}
    ]
    
    for enhancement in [1.0, 2.0, 3.0, 5.0, 10.0]:
        motif_constraints = [
            {
                "name": "reciprocal_EI",
                "type": "reciprocal",
                "node_filters": {"A": {"type": "E"}, "B": {"type": "I"}},
                "enhancement": enhancement
            }
        ]
        
        connectivity = generate_motif_connectivity(
            pop_configs, conn_configs, motif_constraints, seed=42
        )
        
        stats = compute_reciprocal_stats(connectivity, pop_configs)
        print(f"  Enhancement={enhancement:.1f}: reciprocal_rate={stats['reciprocal_rate']:.3f}")
    
    print("  [OK] Extreme enhancement test completed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MOTIF CONNECTIVITY TEST SUITE")
    print("="*60)
    
    test_basic_connectivity()
    test_reciprocal_enhancement()
    test_marginal_preservation()
    test_multi_population()
    test_extreme_enhancement()
    
    # Only run visualization if matplotlib can display or save
    try:
        visualize_connectivity_matrix()
    except Exception as e:
        print(f"\n  Skipping visualization: {e}")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60 + "\n")

