"""
Motif Analysis: Analyze connection motifs between cell types.

This module computes and visualizes the distribution of connection patterns
(no-connection, forward-only, reverse-only, symmetric/reciprocal) between
different cell types, comparing observed patterns to null expectations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import os


def extract_connectivity_matrix(
    synapses_list: List,
    population_configs: List[dict],
    populations: Dict
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]], List[str]]:
    """Extract full connectivity matrix from Brian2 synapse objects."""
    pop_ranges = {}
    pop_order = list(populations.keys())
    current_offset = 0
    
    for name in pop_order:
        n = len(populations[name])
        pop_ranges[name] = (current_offset, current_offset + n)
        current_offset += n
    
    total_neurons = current_offset
    connectivity = np.zeros((total_neurons, total_neurons), dtype=bool)
    
    for syn in synapses_list:
        source_name = syn.source.name
        target_name = syn.target.name
        
        if source_name not in pop_ranges or target_name not in pop_ranges:
            continue
        
        source_offset = pop_ranges[source_name][0]
        target_offset = pop_ranges[target_name][0]
        
        source_indices = np.array(syn.i[:]) + source_offset
        target_indices = np.array(syn.j[:]) + target_offset
        
        connectivity[target_indices, source_indices] = True
    
    return connectivity, pop_ranges, pop_order


def compute_pvalues(
    observed: Dict[str, np.ndarray],
    expected: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Compute p-values for deviation from null hypothesis."""
    pvalues = {}
    
    for motif_type in ['no_connection', 'forward_only', 'reverse_only', 'symmetric']:
        obs = observed[motif_type]
        exp = expected[motif_type]
        total = observed['total_pairs']
        
        pval_matrix = np.zeros_like(obs)
        
        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                n = int(total[i, j])
                k = int(obs[i, j])
                p = exp[i, j] / n if n > 0 else 0.5
                
                if n == 0 or p <= 0 or p >= 1:
                    pval_matrix[i, j] = 0
                    continue
                
                if n > 100:
                    mean = n * p
                    std = np.sqrt(n * p * (1 - p))
                    if std > 0:
                        z = (k - mean) / std
                        pval = 2 * stats.norm.sf(abs(z))
                    else:
                        pval = 1.0
                else:
                    result = stats.binomtest(k, n, p, alternative='two-sided')
                    pval = result.pvalue
                
                if pval < 1e-300:
                    log_pval = 300
                elif pval > 0:
                    log_pval = -np.log10(pval)
                else:
                    log_pval = 0
                
                if k < exp[i, j]:
                    log_pval = -log_pval
                
                pval_matrix[i, j] = log_pval
        
        pvalues[motif_type] = pval_matrix
    
    return pvalues


def plot_motif_analysis(
    observed: Dict[str, np.ndarray],
    expected: Dict[str, np.ndarray],
    type_labels: List[str] = ['E', 'I'],
    title: str = "E/I Symmetric Connection Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 14)
) -> plt.Figure:
    """Create the full motif analysis figure."""
    fig = plt.figure(figsize=figsize)
    
    gs = fig.add_gridspec(5, 4, height_ratios=[0.15, 1, 1, 1, 1], 
                          hspace=0.35, wspace=0.3,
                          left=0.08, right=0.95, top=0.92, bottom=0.05)
    
    motif_types = ['no_connection', 'forward_only', 'reverse_only', 'symmetric']
    motif_titles = ['No-Connection', 'Forward-Only', 'Reverse-Only', 'Symmetric']
    row_labels = ['(Null)', '(Real)', 'Ratio\n(Real/Null)', 'P-values']
    
    ratios = {}
    for mt in motif_types:
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios[mt] = np.where(expected[mt] > 0, 
                                  observed[mt] / expected[mt], 
                                  np.nan)
    
    pvalues = compute_pvalues(observed, expected)
    
    cmap_frac = 'YlOrRd'
    cmap_ratio = 'RdYlGn'
    cmap_pval = 'RdYlGn'
    
    for col, (mt, mt_title) in enumerate(zip(motif_types, motif_titles)):
        ax_icon = fig.add_subplot(gs[0, col])
        _draw_motif_icon(ax_icon, mt)
        ax_icon.set_title(mt_title, fontsize=11, fontweight='bold', pad=10)
    
    for col, mt in enumerate(motif_types):
        ax = fig.add_subplot(gs[1, col])
        total = expected['total_pairs']
        frac = np.where(total > 0, expected[mt] / total, 0)
        _plot_heatmap(ax, frac, expected[mt], type_labels, 
                     f"{motif_titles[col]} {row_labels[0]}", 
                     cmap=cmap_frac, vmin=0, vmax=1, 
                     cbar_label='Fraction' if col == 3 else None)
    
    for col, mt in enumerate(motif_types):
        ax = fig.add_subplot(gs[2, col])
        total = observed['total_pairs']
        frac = np.where(total > 0, observed[mt] / total, 0)
        _plot_heatmap(ax, frac, observed[mt], type_labels,
                     f"{motif_titles[col]} {row_labels[1]}",
                     cmap=cmap_frac, vmin=0, vmax=1,
                     cbar_label='Fraction' if col == 3 else None)
    
    for col, mt in enumerate(motif_types):
        ax = fig.add_subplot(gs[3, col])
        ratio_data = ratios[mt]
        annot = np.empty_like(ratio_data, dtype=object)
        for i in range(ratio_data.shape[0]):
            for j in range(ratio_data.shape[1]):
                r = ratio_data[i, j]
                o = int(observed[mt][i, j])
                e = int(expected[mt][i, j])
                if np.isnan(r):
                    annot[i, j] = f"N/A\n({o}/{e})"
                else:
                    annot[i, j] = f"{r:.2f}\n({o}/{e})"
        
        _plot_heatmap(ax, ratio_data, None, type_labels,
                     f"{motif_titles[col]} {row_labels[2]}",
                     cmap=cmap_ratio, vmin=0, vmax=4, center=1.0,
                     custom_annot=annot,
                     cbar_label='Ratio (Real/Null)' if col == 3 else None)
    
    for col, mt in enumerate(motif_types):
        ax = fig.add_subplot(gs[4, col])
        pval_data = pvalues[mt]
        _plot_heatmap(ax, pval_data, None, type_labels,
                     f"{motif_titles[col]} {row_labels[3]}",
                     cmap=cmap_pval, vmin=-300, vmax=300, center=0,
                     cbar_label='-log10(p-value)' if col == 3 else None,
                     fmt='.0f')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"      [OK] Saved motif analysis: {os.path.basename(save_path)}")
    
    return fig


def _draw_motif_icon(ax, motif_type: str):
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    circle_a = Circle((-0.7, 0), 0.25, facecolor='black', edgecolor='black')
    circle_b = Circle((0.7, 0), 0.25, facecolor='black', edgecolor='black')
    ax.add_patch(circle_a)
    ax.add_patch(circle_b)
    
    ax.text(-0.7, 0, 'A', ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    ax.text(0.7, 0, 'B', ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    
    if motif_type == 'forward_only':
        arrow = FancyArrowPatch((-0.4, 0.05), (0.4, 0.05), arrowstyle='->', mutation_scale=15, color='black', linewidth=2)
        ax.add_patch(arrow)
    elif motif_type == 'reverse_only':
        arrow = FancyArrowPatch((0.4, 0.05), (-0.4, 0.05), arrowstyle='->', mutation_scale=15, color='black', linewidth=2)
        ax.add_patch(arrow)
    elif motif_type == 'symmetric':
        arrow1 = FancyArrowPatch((-0.4, 0.12), (0.4, 0.12), arrowstyle='->', mutation_scale=12, color='black', linewidth=1.5)
        arrow2 = FancyArrowPatch((0.4, -0.12), (-0.4, -0.12), arrowstyle='->', mutation_scale=12, color='black', linewidth=1.5)
        ax.add_patch(arrow1)
        ax.add_patch(arrow2)


def _plot_heatmap(ax, data, counts, labels, title, cmap='YlOrRd', vmin=None, vmax=None, center=None, cbar_label=None, custom_annot=None, fmt='.3f'):
    if center is not None:
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')
    else:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if custom_annot is not None:
                text = custom_annot[i, j]
            elif counts is not None:
                val = data[i, j]
                cnt = int(counts[i, j])
                text = f"{val:{fmt}}\n({cnt})"
            else:
                val = data[i, j]
                text = "N/A" if np.isnan(val) else f"{val:{fmt}}"
            
            bg_val = data[i, j]
            if np.isnan(bg_val):
                color = 'black'
            elif center is not None:
                color = 'black' if abs(bg_val - center) < (vmax - center) * 0.3 else 'white'
            else:
                rel_val = (bg_val - vmin) / (vmax - vmin) if vmin is not None and vmax is not None else 0.5
                color = 'white' if rel_val > 0.5 else 'black'
            
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color=color, fontweight='bold')
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Target', fontsize=9)
    ax.set_ylabel('Source', fontsize=9)
    ax.set_title(title, fontsize=10)
    
    if cbar_label:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(cbar_label, fontsize=8)


def analyze_network_motifs(network, connection_configs, population_configs, group_by_type=True, save_path=None):
    """Analyze motifs in a CellTypeNetwork."""
    connectivity, pop_ranges, pop_order = extract_connectivity_matrix(network.synapses, population_configs, network.populations)
    
    conn_probs = {(c['source'], c['target']): c.get('p', 0.0) for c in connection_configs}
    
    if group_by_type:
        e_pops = [p['name'] for p in population_configs if p.get('type') == 'E']
        i_pops = [p['name'] for p in population_configs if p.get('type') == 'I']
        
        observed = _compute_motifs_aggregated(connectivity, pop_ranges, e_pops, i_pops)
        expected = _compute_expected_aggregated(pop_ranges, conn_probs, e_pops, i_pops)
        
        type_labels = ['E', 'I']
    else:
        type_labels = pop_order
        observed = _compute_motifs_simple(connectivity, pop_ranges, pop_order)
        expected = _compute_expected_simple(pop_ranges, conn_probs, pop_order)
    
    fig = plot_motif_analysis(observed, expected, type_labels, save_path=save_path)
    
    return observed, expected, fig


def _compute_motifs_aggregated(connectivity, pop_ranges, e_pops, i_pops):
    counts = {k: np.zeros((2, 2)) for k in ['no_connection', 'forward_only', 'reverse_only', 'symmetric', 'total_pairs']}
    
    type_pops = [e_pops, i_pops]
    
    for src_type_idx, src_pops in enumerate(type_pops):
        for tgt_type_idx, tgt_pops in enumerate(type_pops):
            for src_pop in src_pops:
                src_start, src_end = pop_ranges[src_pop]
                for tgt_pop in tgt_pops:
                    tgt_start, tgt_end = pop_ranges[tgt_pop]
                    
                    for a in range(src_start, src_end):
                        for b in range(tgt_start, tgt_end):
                            if a == b:
                                continue
                            
                            has_forward = connectivity[b, a]
                            has_reverse = connectivity[a, b]
                            
                            counts['total_pairs'][src_type_idx, tgt_type_idx] += 1
                            
                            if has_forward and has_reverse:
                                counts['symmetric'][src_type_idx, tgt_type_idx] += 1
                            elif has_forward and not has_reverse:
                                counts['forward_only'][src_type_idx, tgt_type_idx] += 1
                            elif not has_forward and has_reverse:
                                counts['reverse_only'][src_type_idx, tgt_type_idx] += 1
                            else:
                                counts['no_connection'][src_type_idx, tgt_type_idx] += 1
    
    return counts


def _compute_expected_aggregated(pop_ranges, conn_probs, e_pops, i_pops):
    expected = {k: np.zeros((2, 2)) for k in ['no_connection', 'forward_only', 'reverse_only', 'symmetric', 'total_pairs']}
    
    type_pops = [e_pops, i_pops]
    
    for src_type_idx, src_pops in enumerate(type_pops):
        for tgt_type_idx, tgt_pops in enumerate(type_pops):
            for src_pop in src_pops:
                n_src = pop_ranges[src_pop][1] - pop_ranges[src_pop][0]
                
                for tgt_pop in tgt_pops:
                    n_tgt = pop_ranges[tgt_pop][1] - pop_ranges[tgt_pop][0]
                    
                    p_forward = conn_probs.get((src_pop, tgt_pop), 0.0)
                    p_reverse = conn_probs.get((tgt_pop, src_pop), 0.0)
                    
                    n_pairs = n_src * (n_tgt - 1) if src_pop == tgt_pop else n_src * n_tgt
                    
                    expected['total_pairs'][src_type_idx, tgt_type_idx] += n_pairs
                    expected['no_connection'][src_type_idx, tgt_type_idx] += n_pairs * (1 - p_forward) * (1 - p_reverse)
                    expected['forward_only'][src_type_idx, tgt_type_idx] += n_pairs * p_forward * (1 - p_reverse)
                    expected['reverse_only'][src_type_idx, tgt_type_idx] += n_pairs * (1 - p_forward) * p_reverse
                    expected['symmetric'][src_type_idx, tgt_type_idx] += n_pairs * p_forward * p_reverse
    
    return expected


def _compute_motifs_simple(connectivity, pop_ranges, pop_order):
    n = len(pop_order)
    counts = {k: np.zeros((n, n)) for k in ['no_connection', 'forward_only', 'reverse_only', 'symmetric', 'total_pairs']}
    
    for i, src_pop in enumerate(pop_order):
        src_start, src_end = pop_ranges[src_pop]
        for j, tgt_pop in enumerate(pop_order):
            tgt_start, tgt_end = pop_ranges[tgt_pop]
            
            for a in range(src_start, src_end):
                for b in range(tgt_start, tgt_end):
                    if a == b:
                        continue
                    
                    has_forward = connectivity[b, a]
                    has_reverse = connectivity[a, b]
                    
                    counts['total_pairs'][i, j] += 1
                    
                    if has_forward and has_reverse:
                        counts['symmetric'][i, j] += 1
                    elif has_forward and not has_reverse:
                        counts['forward_only'][i, j] += 1
                    elif not has_forward and has_reverse:
                        counts['reverse_only'][i, j] += 1
                    else:
                        counts['no_connection'][i, j] += 1
    
    return counts


def _compute_expected_simple(pop_ranges, conn_probs, pop_order):
    n = len(pop_order)
    expected = {k: np.zeros((n, n)) for k in ['no_connection', 'forward_only', 'reverse_only', 'symmetric', 'total_pairs']}
    
    for i, src_pop in enumerate(pop_order):
        n_src = pop_ranges[src_pop][1] - pop_ranges[src_pop][0]
        for j, tgt_pop in enumerate(pop_order):
            n_tgt = pop_ranges[tgt_pop][1] - pop_ranges[tgt_pop][0]
            
            p_forward = conn_probs.get((src_pop, tgt_pop), 0.0)
            p_reverse = conn_probs.get((tgt_pop, src_pop), 0.0)
            
            n_pairs = n_src * (n_tgt - 1) if src_pop == tgt_pop else n_src * n_tgt
            
            expected['total_pairs'][i, j] = n_pairs
            expected['no_connection'][i, j] = n_pairs * (1 - p_forward) * (1 - p_reverse)
            expected['forward_only'][i, j] = n_pairs * p_forward * (1 - p_reverse)
            expected['reverse_only'][i, j] = n_pairs * (1 - p_forward) * p_reverse
            expected['symmetric'][i, j] = n_pairs * p_forward * p_reverse
    
    return expected
