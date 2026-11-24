"""
Motif-Constrained Connectivity Generation

This module provides tools to generate connectivity matrices with specified motif
overrepresentation while maintaining target marginal connection probabilities.

Supported Motif Types:
- reciprocal: A→B and B→A both exist (bidirectional connection)
- chain: A→B→C (sequential path)
- convergent: A→C and B→C (multiple inputs to same target)
- divergent: A→B and A→C (one source, multiple targets)

Enhancement Factor:
- enhancement=1.0 means independent (no motif bias)
- enhancement=3.0 means the motif is 3x more likely than chance

Configuration Format:
{
    "motif_constraints": [
        {
            "name": "reciprocal_EI",
            "type": "reciprocal",
            "node_filters": {
                "A": {"type": "E"},           # Filter by cell type
                "B": {"type": "I"}
            },
            # OR specify populations directly:
            # "populations": {
            #     "A": ["L1_E", "L2_E"],
            #     "B": ["L1_I", "L2_I"]
            # },
            "enhancement": 3.0
        }
    ]
}
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class MotifConstraint:
    """Represents a single motif constraint."""
    name: str
    motif_type: str  # 'reciprocal', 'chain', 'convergent', 'divergent'
    node_filters: Dict[str, Dict[str, Any]]  # e.g., {"A": {"type": "E"}, "B": {"type": "I"}}
    populations: Optional[Dict[str, List[str]]]  # Direct population specification
    enhancement: float
    
    @classmethod
    def from_config(cls, config: dict) -> 'MotifConstraint':
        return cls(
            name=config.get('name', 'unnamed'),
            motif_type=config['type'],
            node_filters=config.get('node_filters', {}),
            populations=config.get('populations'),
            enhancement=config.get('enhancement', 1.0)
        )


class MotifConnectivityGenerator:
    """
    Generates connectivity matrices with motif constraints.
    
    Usage:
        generator = MotifConnectivityGenerator(
            population_configs=pop_configs,
            connection_configs=conn_configs,
            motif_constraints=motif_configs
        )
        connectivity = generator.generate()
        # connectivity[(source_pop, target_pop)] = list of (source_idx, target_idx) tuples
    """
    
    def __init__(
        self,
        population_configs: List[dict],
        connection_configs: List[dict],
        motif_constraints: Optional[List[dict]] = None,
        seed: Optional[int] = None
    ):
        self.population_configs = population_configs
        self.connection_configs = connection_configs
        self.motif_constraints = [
            MotifConstraint.from_config(mc) for mc in (motif_constraints or [])
        ]
        
        if seed is not None:
            np.random.seed(seed)
        
        # Build population info
        self._build_population_info()
        
        # Build connection probability matrix
        self._build_connection_probs()
    
    def _build_population_info(self):
        """Extract population metadata."""
        self.pop_info = {}
        for config in self.population_configs:
            name = config['name']
            self.pop_info[name] = {
                'n': config['n'],
                'type': config.get('type', 'E'),  # E or I
                'config': config
            }
        self.pop_names = list(self.pop_info.keys())
    
    def _build_connection_probs(self):
        """Build matrix of connection probabilities between populations."""
        self.conn_probs = {}  # (source, target) -> probability
        self.conn_configs_map = {}  # (source, target) -> full config
        
        for config in self.connection_configs:
            source = config['source']
            target = config['target']
            # Skip if source is input layer (not in population_configs)
            if source not in self.pop_info:
                continue
            if target not in self.pop_info:
                continue
            self.conn_probs[(source, target)] = config.get('p', 0.0)
            self.conn_configs_map[(source, target)] = config
    
    def _get_populations_for_node(self, node_label: str, constraint: MotifConstraint) -> List[str]:
        """
        Get list of population names that match a node label in a motif.
        
        Args:
            node_label: e.g., "A" or "B"
            constraint: The motif constraint containing filters
        
        Returns:
            List of population names matching the filter
        """
        # If populations are directly specified, use them
        if constraint.populations and node_label in constraint.populations:
            return constraint.populations[node_label]
        
        # Otherwise, use filters
        if node_label not in constraint.node_filters:
            # No filter means all populations
            return self.pop_names
        
        filter_spec = constraint.node_filters[node_label]
        matching_pops = []
        
        for pop_name, pop_info in self.pop_info.items():
            matches = True
            
            # Check type filter
            if 'type' in filter_spec:
                if pop_info['type'] != filter_spec['type']:
                    matches = False
            
            # Check layer filter (if population name contains layer info)
            if 'layer' in filter_spec:
                if filter_spec['layer'] not in pop_name:
                    matches = False
            
            # Check name pattern
            if 'name_contains' in filter_spec:
                if filter_spec['name_contains'] not in pop_name:
                    matches = False
            
            # Check explicit population list
            if 'populations' in filter_spec:
                if pop_name not in filter_spec['populations']:
                    matches = False
            
            if matches:
                matching_pops.append(pop_name)
        
        return matching_pops
    
    def generate(self) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
        """
        Generate connectivity with motif constraints.
        
        Returns:
            Dictionary mapping (source_pop, target_pop) to list of (source_idx, target_idx) connections
        """
        # Initialize empty connectivity
        connectivity = defaultdict(list)
        
        # Group constraints by type for processing order
        reciprocal_constraints = [c for c in self.motif_constraints if c.motif_type == 'reciprocal']
        chain_constraints = [c for c in self.motif_constraints if c.motif_type == 'chain']
        # Other types can be added later
        
        # First, generate base random connectivity for all pairs
        base_connectivity = self._generate_base_connectivity()
        
        # Apply reciprocal constraints
        if reciprocal_constraints:
            connectivity = self._apply_reciprocal_constraints(
                base_connectivity, reciprocal_constraints
            )
        else:
            # Convert base connectivity matrices to list format
            connectivity = self._matrices_to_lists(base_connectivity)
        
        # Apply chain constraints
        if chain_constraints:
            connectivity = self._apply_chain_constraints(connectivity, chain_constraints)
        
        return dict(connectivity)
    
    def _matrices_to_lists(
        self,
        connectivity: Dict[Tuple[str, str], np.ndarray]
    ) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
        """Convert connectivity matrices to list of (i, j) tuples."""
        result = {}
        for key, matrix in connectivity.items():
            indices = np.where(matrix)
            result[key] = list(zip(indices[0].tolist(), indices[1].tolist()))
        return result
    
    def _generate_base_connectivity(self) -> Dict[Tuple[str, str], np.ndarray]:
        """
        Generate base connectivity matrices (before motif adjustments).
        
        Returns:
            Dict mapping (source, target) to boolean connectivity matrix
        """
        connectivity = {}
        
        for (source, target), prob in self.conn_probs.items():
            n_source = self.pop_info[source]['n']
            n_target = self.pop_info[target]['n']
            
            # Generate random connectivity matrix
            conn_matrix = np.random.random((n_source, n_target)) < prob
            connectivity[(source, target)] = conn_matrix
        
        return connectivity
    
    def _apply_reciprocal_constraints(
        self,
        connectivity: Dict[Tuple[str, str], np.ndarray],
        constraints: List[MotifConstraint]
    ) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
        """
        Apply reciprocal motif constraints.
        
        For reciprocal motifs, we need to ensure that:
        - P(B→A | A→B) = min(1, r * p_BA)
        - P(B→A | ¬A→B) is adjusted to maintain marginal P(B→A) = p_BA
        
        The adjustment formula:
        P(B→A | ¬A→B) = (p_BA - r * p_BA * p_AB) / (1 - p_AB)
                      = p_BA * (1 - r * p_AB) / (1 - p_AB)
        """
        # Convert matrices to list format
        result = defaultdict(list)
        
        # Track which pairs are affected by reciprocal constraints
        reciprocal_pairs = set()
        
        for constraint in constraints:
            pops_A = self._get_populations_for_node('A', constraint)
            pops_B = self._get_populations_for_node('B', constraint)
            r = constraint.enhancement
            
            for pop_A in pops_A:
                for pop_B in pops_B:
                    if pop_A == pop_B:
                        # Self-connections within same population (recurrent)
                        self._apply_reciprocal_same_population(
                            connectivity, result, pop_A, r
                        )
                        reciprocal_pairs.add((pop_A, pop_A))
                    else:
                        # Connections between different populations
                        self._apply_reciprocal_different_populations(
                            connectivity, result, pop_A, pop_B, r
                        )
                        reciprocal_pairs.add((pop_A, pop_B))
                        reciprocal_pairs.add((pop_B, pop_A))
        
        # Add non-affected connections as-is
        for (source, target), conn_matrix in connectivity.items():
            if (source, target) not in reciprocal_pairs:
                indices = np.where(conn_matrix)
                result[(source, target)] = list(zip(indices[0].tolist(), indices[1].tolist()))
        
        return result
    
    def _apply_reciprocal_same_population(
        self,
        connectivity: Dict[Tuple[str, str], np.ndarray],
        result: Dict[Tuple[str, str], List[Tuple[int, int]]],
        pop: str,
        enhancement: float
    ):
        """Apply reciprocal constraint within same population (recurrent connections)."""
        key = (pop, pop)
        if key not in connectivity:
            return
        
        n = self.pop_info[pop]['n']
        p = self.conn_probs.get(key, 0.0)
        
        if p == 0:
            return
        
        # For same population, we need to handle i→j and j→i together
        conn_matrix = np.zeros((n, n), dtype=bool)
        
        # Calculate adjusted probabilities
        p_recip_given_exists = min(1.0, enhancement * p)
        
        # P(j→i | ¬i→j) adjusted to maintain marginal
        if p < 1.0:
            p_adjusted = p * (1 - enhancement * p) / (1 - p)
            p_adjusted = max(0.0, min(1.0, p_adjusted))
        else:
            p_adjusted = 0.0
        
        # Process each unordered pair once
        for i in range(n):
            for j in range(i + 1, n):
                # Decide i→j first with base probability
                has_ij = np.random.random() < p
                
                if has_ij:
                    # j→i with enhanced probability
                    has_ji = np.random.random() < p_recip_given_exists
                else:
                    # j→i with adjusted probability
                    has_ji = np.random.random() < p_adjusted
                
                conn_matrix[i, j] = has_ij
                conn_matrix[j, i] = has_ji
        
        # Convert to list
        indices = np.where(conn_matrix)
        result[key] = list(zip(indices[0].tolist(), indices[1].tolist()))
    
    def _apply_reciprocal_different_populations(
        self,
        connectivity: Dict[Tuple[str, str], np.ndarray],
        result: Dict[Tuple[str, str], List[Tuple[int, int]]],
        pop_A: str,
        pop_B: str,
        enhancement: float
    ):
        """Apply reciprocal constraint between two different populations."""
        key_AB = (pop_A, pop_B)
        key_BA = (pop_B, pop_A)
        
        p_AB = self.conn_probs.get(key_AB, 0.0)
        p_BA = self.conn_probs.get(key_BA, 0.0)
        
        n_A = self.pop_info[pop_A]['n']
        n_B = self.pop_info[pop_B]['n']
        
        # Generate A→B connections first
        conn_AB = np.random.random((n_A, n_B)) < p_AB
        
        # For B→A, use conditional probabilities
        # P(B→A | A→B) = min(1, r * p_BA)
        # P(B→A | ¬A→B) = p_BA * (1 - r * p_AB) / (1 - p_AB)  [adjusted]
        
        p_BA_given_AB = min(1.0, enhancement * p_BA)
        
        if p_AB < 1.0:
            numerator = p_BA * (1.0 - enhancement * p_AB)
            denominator = 1.0 - p_AB
            p_BA_given_not_AB = max(0.0, min(1.0, numerator / denominator))
        else:
            p_BA_given_not_AB = 0.0
        
        # Generate B→A based on A→B
        conn_BA = np.zeros((n_B, n_A), dtype=bool)
        
        for i in range(n_A):
            for j in range(n_B):
                if conn_AB[i, j]:
                    # A→B exists, use enhanced probability for B→A
                    conn_BA[j, i] = np.random.random() < p_BA_given_AB
                else:
                    # A→B doesn't exist, use adjusted probability
                    conn_BA[j, i] = np.random.random() < p_BA_given_not_AB
        
        # Convert to lists
        indices_AB = np.where(conn_AB)
        result[key_AB] = list(zip(indices_AB[0].tolist(), indices_AB[1].tolist()))
        
        indices_BA = np.where(conn_BA)
        result[key_BA] = list(zip(indices_BA[0].tolist(), indices_BA[1].tolist()))
    
    def _apply_chain_constraints(
        self,
        connectivity: Dict[Tuple[str, str], List[Tuple[int, int]]],
        constraints: List[MotifConstraint]
    ) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
        """
        Apply chain motif constraints (A→B→C).
        
        For chains, we enhance the probability of B→C given that A→B exists.
        This is more complex because we need to track which B neurons receive A input.
        """
        # For now, chain constraints require explicit population specification
        # in the 'sequence' field of the constraint config
        
        for constraint in constraints:
            if 'sequence' not in constraint.node_filters:
                # Need sequence specification for chains
                continue
            
            sequence = constraint.node_filters['sequence']  # e.g., ['L5_E', 'L1_I', 'L2_E']
            r = constraint.enhancement
            
            if len(sequence) < 3:
                continue
            
            # Apply enhancement along the chain
            for i in range(len(sequence) - 2):
                pop_A, pop_B, pop_C = sequence[i], sequence[i+1], sequence[i+2]
                
                connectivity = self._enhance_chain_step(
                    connectivity, pop_A, pop_B, pop_C, r
                )
        
        return connectivity
    
    def _enhance_chain_step(
        self,
        connectivity: Dict[Tuple[str, str], List[Tuple[int, int]]],
        pop_A: str,
        pop_B: str,
        pop_C: str,
        enhancement: float
    ) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
        """
        Enhance B→C probability for B neurons that receive A→B connections.
        """
        key_AB = (pop_A, pop_B)
        key_BC = (pop_B, pop_C)
        
        if key_AB not in connectivity or key_BC not in self.conn_probs:
            return connectivity
        
        # Find B neurons that receive input from A
        connections_AB = connectivity.get(key_AB, [])
        B_neurons_with_A_input = set(conn[1] for conn in connections_AB)
        
        n_B = self.pop_info[pop_B]['n']
        n_C = self.pop_info[pop_C]['n']
        p_BC = self.conn_probs.get(key_BC, 0.0)
        
        # Calculate adjusted probabilities
        # For B neurons with A input: enhanced probability
        # For B neurons without A input: adjusted to maintain marginal
        
        p_BC_enhanced = min(1.0, enhancement * p_BC)
        
        frac_B_with_A = len(B_neurons_with_A_input) / n_B if n_B > 0 else 0
        
        if frac_B_with_A < 1.0:
            numerator = p_BC - p_BC_enhanced * frac_B_with_A
            denominator = 1.0 - frac_B_with_A
            p_BC_adjusted = max(0.0, min(1.0, numerator / denominator))
        else:
            p_BC_adjusted = p_BC_enhanced
        
        # Regenerate B→C connections
        new_connections_BC = []
        for b in range(n_B):
            prob = p_BC_enhanced if b in B_neurons_with_A_input else p_BC_adjusted
            for c in range(n_C):
                if np.random.random() < prob:
                    new_connections_BC.append((b, c))
        
        connectivity[key_BC] = new_connections_BC
        return connectivity
    
    def get_connectivity_stats(
        self,
        connectivity: Dict[Tuple[str, str], List[Tuple[int, int]]]
    ) -> Dict[str, Any]:
        """
        Compute statistics about the generated connectivity.
        
        Returns:
            Dictionary with connection counts, actual probabilities, motif counts
        """
        stats = {
            'connection_counts': {},
            'actual_probabilities': {},
            'target_probabilities': {},
            'reciprocal_counts': {},
            'reciprocal_expected': {}
        }
        
        for (source, target), connections in connectivity.items():
            n_source = self.pop_info.get(source, {}).get('n', 0)
            n_target = self.pop_info.get(target, {}).get('n', 0)
            
            if n_source == 0 or n_target == 0:
                continue
            
            n_possible = n_source * n_target
            if source == target:
                n_possible = n_source * (n_target - 1)  # No self-connections
            
            actual_p = len(connections) / n_possible if n_possible > 0 else 0
            
            stats['connection_counts'][(source, target)] = len(connections)
            stats['actual_probabilities'][(source, target)] = actual_p
            stats['target_probabilities'][(source, target)] = self.conn_probs.get((source, target), 0)
        
        # Compute reciprocal statistics
        for (pop_A, pop_B), conns_AB in connectivity.items():
            if pop_A >= pop_B:  # Only count each pair once
                continue
            
            conns_BA = connectivity.get((pop_B, pop_A), [])
            
            # Convert to sets for fast lookup
            set_AB = set(conns_AB)
            set_BA = set((j, i) for (i, j) in conns_BA)  # Flip indices
            
            # Count reciprocal connections
            reciprocal = len(set_AB & set_BA)
            
            # Expected under independence
            p_AB = stats['actual_probabilities'].get((pop_A, pop_B), 0)
            p_BA = stats['actual_probabilities'].get((pop_B, pop_A), 0)
            
            n_A = self.pop_info.get(pop_A, {}).get('n', 0)
            n_B = self.pop_info.get(pop_B, {}).get('n', 0)
            
            expected = p_AB * p_BA * n_A * n_B
            
            stats['reciprocal_counts'][(pop_A, pop_B)] = reciprocal
            stats['reciprocal_expected'][(pop_A, pop_B)] = expected
        
        return stats


def validate_motif_constraints(
    connection_configs: List[dict],
    motif_constraints: List[dict]
) -> List[str]:
    """
    Validate that motif constraints are satisfiable given connection probabilities.
    
    Returns:
        List of warning/error messages (empty if all OK)
    """
    warnings = []
    
    conn_probs = {(c['source'], c['target']): c.get('p', 0.0) for c in connection_configs}
    
    for mc in motif_constraints:
        if mc['type'] == 'reciprocal':
            r = mc.get('enhancement', 1.0)
            
            # For reciprocal constraints, r must be <= 1/max(p_AB, p_BA) for each pair
            # This is checked during generation, but we can warn about extreme values
            if r > 10:
                warnings.append(
                    f"Motif '{mc.get('name', 'unnamed')}' has very high enhancement ({r}). "
                    "This may not be achievable for all population pairs."
                )
    
    return warnings


# Convenience function for integration with existing network code
def generate_motif_connectivity(
    population_configs: List[dict],
    connection_configs: List[dict],
    motif_constraints: Optional[List[dict]] = None,
    seed: Optional[int] = None
) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
    """
    Generate connectivity with motif constraints.
    
    This is the main entry point for the motif connectivity system.
    
    Args:
        population_configs: List of population configuration dicts
        connection_configs: List of connection configuration dicts  
        motif_constraints: Optional list of motif constraint configs
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping (source_pop, target_pop) to list of (i, j) connections
    """
    generator = MotifConnectivityGenerator(
        population_configs=population_configs,
        connection_configs=connection_configs,
        motif_constraints=motif_constraints,
        seed=seed
    )
    return generator.generate()

