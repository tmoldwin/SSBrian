"""
Network Architecture Visualization

Generates a diagram of the neural network based on a config JSON file.
- Excitatory (E) populations: red circles
- Inhibitory (I) populations: blue circles  
- Input layers: green circles
- Arrows show connectivity with width proportional to connection probability
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.collections import PatchCollection


def load_config(config_path):
    """Load config from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_population_color(pop_type):
    """Return color based on population type."""
    colors = {
        'E': '#E74C3C',      # Red for excitatory
        'I': '#3498DB',      # Blue for inhibitory
        'input': '#2ECC71',  # Green for input
    }
    return colors.get(pop_type, '#95A5A6')  # Gray default


def get_connection_color(source_type):
    """Return arrow color based on source type (excitatory vs inhibitory)."""
    if source_type == 'I':
        return '#3498DB'  # Blue for inhibitory connections
    elif source_type == 'E':
        return '#E74C3C'  # Red for excitatory connections
    else:
        return '#2ECC71'  # Green for input connections


def calculate_layout(populations, input_layer):
    """
    Calculate positions for each population in a vertical layered layout.
    Layer0 (input) at top, subsequent layers below.
    Returns dict mapping population name to (x, y) position and layer boundaries.
    """
    positions = {}
    layer_y_positions = {}  # Track y position of each layer for drawing separators
    
    # Group populations by layer (extract layer number from name like L1_E, L2_I)
    layers = {}
    for pop in populations:
        name = pop['name']
        # Try to extract layer number
        if name.startswith('L'):
            try:
                layer_num = int(name[1])
            except (ValueError, IndexError):
                layer_num = 1
        else:
            layer_num = 1
        
        if layer_num not in layers:
            layers[layer_num] = []
        layers[layer_num].append(pop)
    
    # Input layer at top (y=0, will be highest)
    positions[input_layer['name']] = (0, 0)
    layer_y_positions[0] = 0  # Input layer is "layer 0"
    
    # Position each layer vertically (going down)
    layer_nums = sorted(layers.keys())
    for i, layer_num in enumerate(layer_nums):
        y = -(i + 1) * 2  # Negative y = going down
        layer_y_positions[layer_num] = y
        pops_in_layer = layers[layer_num]
        
        # Spread populations horizontally within the layer
        n_pops = len(pops_in_layer)
        for j, pop in enumerate(pops_in_layer):
            # Center horizontally
            x = (j - (n_pops - 1) / 2) * 1.8
            positions[pop['name']] = (x, y)
    
    return positions, layer_y_positions


def draw_self_connection(ax, pos, radius, color, linewidth, prob=None, offset_angle=45):
    """
    Draw a self-loop arrow that starts and ends on the node's edge.
    
    Parameters
    ----------
    ax : matplotlib axes
    pos : tuple (x, y) center of the node
    radius : float, radius of the node circle
    color : str, arrow color
    linewidth : float, arrow line width
    prob : float, optional, connection probability to display as label
    offset_angle : float, angle in degrees for loop position (default 45 = upper right)
    """
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path
    import matplotlib.patches as patches
    
    x, y = pos
    
    # The loop will be centered at offset_angle, spanning on the circle
    center_angle = np.radians(offset_angle)
    spread = np.radians(45)  # How far apart the start/end points are
    
    # Start and end points on the circle's edge
    start_angle = center_angle - spread
    end_angle = center_angle + spread
    
    start_x = x + radius * np.cos(start_angle)
    start_y = y + radius * np.sin(start_angle)
    end_x = x + radius * np.cos(end_angle)
    end_y = y + radius * np.sin(end_angle)
    
    # Control point for the curve - positioned outward from the circle
    bulge = radius * 1.5  # How far the loop extends outward
    ctrl_x = x + (radius + bulge) * np.cos(center_angle)
    ctrl_y = y + (radius + bulge) * np.sin(center_angle)
    
    # Draw curved arrow using FancyArrowPatch with a curved connection
    arrow = FancyArrowPatch(
        (start_x, start_y), (end_x, end_y),
        connectionstyle=f"arc3,rad=0.8",  # Larger rad = more curved/bigger loop
        arrowstyle='-|>',
        color=color,
        linewidth=linewidth,
        mutation_scale=18,
        zorder=5
    )
    ax.add_patch(arrow)
    
    # Add probability label if provided
    if prob is not None:
        # Position label at the apex of the curve
        label_distance = radius + bulge * 0.6
        label_x = x + label_distance * np.cos(center_angle)
        label_y = y + label_distance * np.sin(center_angle)
        ax.text(label_x, label_y, f'{prob:.0%}',
                fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                         edgecolor='none', alpha=0.8),
                zorder=7)


def draw_connection(ax, src_pos, tgt_pos, src_radius, tgt_radius, color, linewidth, 
                    prob, curve_rad=0, label_offset=0):
    """
    Draw a curved arrow connection between two nodes, touching edges.
    
    Parameters
    ----------
    ax : matplotlib axes
    src_pos, tgt_pos : array-like, (x, y) positions
    src_radius, tgt_radius : float, node radii
    color : str
    linewidth : float
    prob : float, connection probability for label
    curve_rad : float, curvature (positive=left, negative=right, 0=straight)
    label_offset : float, perpendicular offset for label position
    """
    src_pos = np.array(src_pos)
    tgt_pos = np.array(tgt_pos)
    
    # Direction from source to target
    direction = tgt_pos - src_pos
    distance = np.linalg.norm(direction)
    if distance == 0:
        return
    direction = direction / distance
    
    # Perpendicular vector (for curved arrows)
    perp = np.array([-direction[1], direction[0]])
    
    # For curved arrows, adjust start/end points along the curve
    if curve_rad != 0:
        # Offset the connection points slightly along perpendicular
        offset = perp * abs(curve_rad) * 0.3
        if curve_rad > 0:
            start = src_pos + direction * src_radius + offset * 0.5
            end = tgt_pos - direction * tgt_radius + offset * 0.5
        else:
            start = src_pos + direction * src_radius - offset * 0.5
            end = tgt_pos - direction * tgt_radius - offset * 0.5
    else:
        start = src_pos + direction * src_radius
        end = tgt_pos - direction * tgt_radius
    
    # Draw arrow using FancyArrowPatch for clean edges
    if curve_rad != 0:
        arrow = FancyArrowPatch(
            tuple(start), tuple(end),
            connectionstyle=f'arc3,rad={curve_rad}',
            arrowstyle='-|>',
            color=color,
            linewidth=linewidth,
            mutation_scale=14,
            zorder=3
        )
    else:
        arrow = FancyArrowPatch(
            tuple(start), tuple(end),
            arrowstyle='-|>',
            color=color,
            linewidth=linewidth,
            mutation_scale=14,
            zorder=3
        )
    ax.add_patch(arrow)
    
    # Calculate label position along the curve
    mid = (start + end) / 2
    if curve_rad != 0:
        # Offset label along the curve
        mid = mid + perp * curve_rad * 0.8
    if label_offset != 0:
        mid = mid + perp * label_offset
    
    ax.text(mid[0], mid[1], f'{prob:.0%}',
            fontsize=7, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                     edgecolor='none', alpha=0.85),
            zorder=8)


def draw_network_diagram(config, output_path=None, figsize=(12, 14)):
    """
    Draw the network architecture diagram (vertical layout, Layer0 on top).
    
    Parameters
    ----------
    config : dict
        Network configuration dictionary
    output_path : str, optional
        Path to save the figure. If None, displays interactively.
    figsize : tuple
        Figure size in inches
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract components
    input_layer = config.get('input_layer', {'name': 'Input', 'n': 100})
    populations = config.get('populations', [])
    connections = config.get('connections', [])
    
    # Build population type lookup
    pop_types = {input_layer['name']: 'input'}
    pop_sizes = {input_layer['name']: input_layer.get('n', 100)}
    for pop in populations:
        pop_types[pop['name']] = pop.get('type', 'E')
        pop_sizes[pop['name']] = pop.get('n', 100)
    
    # Calculate layout (now returns positions and layer boundaries)
    positions, layer_y_positions = calculate_layout(populations, input_layer)
    
    # Determine node sizes based on neuron count
    max_n = max(pop_sizes.values())
    min_radius = 0.25
    max_radius = 0.40
    
    def get_radius(n):
        return min_radius + (max_radius - min_radius) * (n / max_n)
    
    # Get x extent for drawing layer separators
    all_x = [pos[0] for pos in positions.values()]
    x_min, x_max = min(all_x) - 1.5, max(all_x) + 1.5
    
    # Draw horizontal dashed lines between layers
    sorted_layers = sorted(layer_y_positions.keys())
    for i in range(len(sorted_layers) - 1):
        y1 = layer_y_positions[sorted_layers[i]]
        y2 = layer_y_positions[sorted_layers[i + 1]]
        separator_y = (y1 + y2) / 2
        ax.axhline(separator_y, color='gray', linestyle='--', linewidth=1.5, 
                   alpha=0.6, zorder=1, xmin=0.05, xmax=0.95)
    
    # Build connection pair tracking for curve directions
    connection_pairs = {}
    for conn in connections:
        src, tgt = conn['source'], conn['target']
        if src != tgt:
            key = tuple(sorted([src, tgt]))
            if key not in connection_pairs:
                connection_pairs[key] = []
            connection_pairs[key].append((src, tgt))
    
    # Track self-connection count per node for offset positioning
    self_conn_count = {}
    
    # Draw connections first (so they're behind nodes)
    for conn in connections:
        source = conn['source']
        target = conn['target']
        prob = conn.get('p', 1.0)
        
        if source not in positions or target not in positions:
            continue
        
        source_type = pop_types.get(source, 'E')
        color = get_connection_color(source_type)
        linewidth = 1.5 + prob * 2  # Scale line width by probability
        
        if source == target:
            # Self-connection (recurrent) - use horizontal angles for vertical layout
            count = self_conn_count.get(source, 0)
            # Alternate between left (-135°) and right (45°) for vertical layout
            offset_angle = -45 if count % 2 == 0 else -135
            self_conn_count[source] = count + 1
            
            draw_self_connection(ax, positions[source], 
                               get_radius(pop_sizes[source]), 
                               color, linewidth, prob=prob, 
                               offset_angle=offset_angle)
        else:
            # Connection between different populations
            src_pos = positions[source]
            tgt_pos = positions[target]
            src_radius = get_radius(pop_sizes[source])
            tgt_radius = get_radius(pop_sizes[target])
            
            # Check if bidirectional
            key = tuple(sorted([source, target]))
            is_bidirectional = len(connection_pairs.get(key, [])) > 1
            
            if is_bidirectional:
                # Curve arrows to avoid overlap
                if source < target:
                    curve_rad = 0.25
                else:
                    curve_rad = -0.25
            else:
                curve_rad = 0
            
            draw_connection(ax, src_pos, tgt_pos, src_radius, tgt_radius,
                          color, linewidth, prob, curve_rad=curve_rad)
    
    # Draw nodes (on top of connections)
    for name, pos in positions.items():
        pop_type = pop_types.get(name, 'E')
        color = get_population_color(pop_type)
        n = pop_sizes.get(name, 100)
        radius = get_radius(n)
        
        circle = Circle(pos, radius, facecolor=color, edgecolor='black',
                       linewidth=2, alpha=0.95, zorder=10)
        ax.add_patch(circle)
        
        # Add label
        ax.text(pos[0], pos[1], f'{name}\n(n={n})',
               ha='center', va='center', fontsize=9, fontweight='bold',
               color='white', zorder=11)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor=get_population_color('input'), 
                      edgecolor='black', label='Input Layer'),
        mpatches.Patch(facecolor=get_population_color('E'), 
                      edgecolor='black', label='Excitatory (E)'),
        mpatches.Patch(facecolor=get_population_color('I'), 
                      edgecolor='black', label='Inhibitory (I)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # Set title
    config_name = config.get('name', 'Network')
    ax.set_title(f'Network Architecture: {config_name}', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Clean up axes
    ax.set_aspect('equal')
    ax.autoscale()
    margin = 0.8
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[0] - margin, xlim[1] + margin)
    ax.set_ylim(ylim[0] - margin, ylim[1] + margin)
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Network diagram saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Generate a network architecture diagram from a config file.'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to config JSON file, or config name (e.g., test_20_stims)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for the diagram (default: saves to Figures folder)'
    )
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[12, 8],
        help='Figure size in inches (width height)'
    )
    
    args = parser.parse_args()
    
    # Resolve config path
    config_path = args.config
    if not os.path.exists(config_path):
        # Try looking in configs/runs
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'configs', 'runs', f'{args.config}.json')
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {args.config}")
        return
    
    config = load_config(config_path)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_name = config.get('name', 'network')
        output_path = os.path.join(script_dir, 'Runs', 'single_runs', 
                                   config_name, 'latest', 'Figures', 
                                   'network_architecture.png')
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    draw_network_diagram(config, output_path, figsize=tuple(args.figsize))


if __name__ == '__main__':
    main()


