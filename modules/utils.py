from typing import Dict, List, Optional, Tuple

import pandas as pd
import json
import seaborn as sns
import os
from scipy import stats

import numpy as np
import torch
from torch_geometric.data import Data

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import networkx as nx
from scipy.spatial import ConvexHull

from modules.hypergraph import Hypergraph

LISTENER_FORWARD_MAPPING = {
    1: [0, 0, 0, 0],
    2: [0, 0, 0, 1],
    3: [0, 0, 1, 1],
    4: [0, 0, 1, 0],
    5: [0, 1, 1, 0],
    6: [0, 1, 1, 1],
    7: [0, 1, 0, 1],
    8: [0, 1, 0, 0],
    9: [1, 1, 0, 0],
    10: [1, 1, 0, 1],
    11: [1, 1, 1, 1]
}

LISTENER_BACKWARD_MAPPING = {
    (0, 0, 0, 0): 1,
    (0, 0, 0, 1): 2,
    (0, 0, 1, 1): 3,
    (0, 0, 1, 0): 4,
    (0, 1, 1, 0): 5,
    (0, 1, 1, 1): 6,
    (0, 1, 0, 1): 7,
    (0, 1, 0, 0): 8,
    (1, 1, 0, 0): 9,
    (1, 1, 0, 1): 10,
    (1, 1, 1, 1): 11
}

SPEAKER_FORWARD_MAPPING = {
    1: [0, 0, 0, 0],
    2: [0, 0, 0, 1],
    3: [0, 0, 1, 1],
}

SPEAKER_BACKWARD_MAPPING = {
    (0, 0, 0, 0): 1,
    (0, 0, 0, 1): 2,
    (0, 0, 1, 1): 3,
}

def prepare_listener(listener_obs, shuffle=False):
    listener_obs_mapped = torch.zeros(len(listener_obs), 5)
    listener_obs_mapped[:, :4] = torch.tensor(list(LISTENER_FORWARD_MAPPING.values())[:len(listener_obs)])
    listener_obs_mapped[0:len(listener_obs), 4] = torch.tensor(listener_obs)
    if shuffle:
        return listener_obs_mapped[torch.randperm(listener_obs_mapped.size(0))]
    else:
        return listener_obs_mapped


def prepare_speaker(speaker_obs, vectorizer, filter, shuffle=False):
    r, g, b = speaker_obs[0], speaker_obs[1], speaker_obs[2]

    # Use the extractor to get the structured data, including token_vectors
    token_vectors = vectorizer.process_rgb_to_embeddings(r, g, b)

    # Filter the tokens based on the provided list
    filtered_vectors = {}
    for token, vector in token_vectors.items():
        if any(key_part in token for key_part in filter):
            filtered_vectors[token] = vector

    if not filtered_vectors:
        print("Warning: No matching tokens found. Returning empty tensor.")
        embed_dim = vectorizer.model.config.hidden_size
        return torch.zeros((0, 4 + embed_dim), dtype=torch.float32)

    num_tokens = len(filtered_vectors)
    embed_dim = vectorizer.model.config.hidden_size

    speaker_obs_mapped = torch.zeros(num_tokens, 4 + embed_dim)

    vecs = torch.tensor(np.stack(list(filtered_vectors.values())), dtype=torch.float32)

    speaker_obs_mapped[:, :4] = torch.tensor(list(SPEAKER_FORWARD_MAPPING.values())[:num_tokens])
    speaker_obs_mapped[:, 4:] = vecs

    if shuffle:
        return speaker_obs_mapped[torch.randperm(speaker_obs_mapped.size(0))]
    else:
        return speaker_obs_mapped


def shift_graph(graph, x, y):
    offset = torch.tensor([x, y], dtype=graph.pos.dtype)
    graph.pos = graph.pos + offset
    return graph


def prepare_hypergraph(components):
    hg_constrained = Hypergraph.build_relaxed(
        data_list=components,
        s = 4,
        rho = 2.0,
        force_connect=True
    )
    return hg_constrained


def create_sample_components(
        n_components: int = 4,
        nodes_per_comp: int = 10,
        feature_dim: int = 8
) -> List[Data]:

    data_list = []

    for i in range(n_components):
        # Create node features
        x = torch.randn((nodes_per_comp, feature_dim))

        # Create positions in 2D space
        center = torch.tensor([i * 3.0, (i % 2) * 3.0])
        pos = center + torch.randn((nodes_per_comp, 2))

        # Create edges (chain + some random connections)
        edges = []
        # Chain backbone
        for j in range(nodes_per_comp - 1):
            edges.extend([[j, j + 1], [j + 1, j]])

        # Add some random edges
        for _ in range(nodes_per_comp // 3):
            a, b = np.random.choice(nodes_per_comp, 2, replace=False)
            if a != b:
                edges.extend([[a, b], [b, a]])

        edge_index = torch.tensor(edges, dtype=torch.long).T

        # Compute edge attributes (distances)
        src, dst = edge_index
        edge_attr = torch.norm(pos[src] - pos[dst], dim=1, keepdim=True)

        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos
        )

        data_list.append(data)

    return data_list


def visualize_hypergraph(
        data: Data,
        title: str = "Hypergraph Visualization",
        figsize: Tuple[float, float] = (12, 10),
        save_path: Optional[str] = None,
        show_hyperedge_hulls: bool = True,
        node_size: int = 100,
        show_legend: bool = True
):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Get positions and component labels
    if data.pos is not None:
        pos = data.pos.numpy()
    else:
        # Create layout using networkx if no positions
        G = nx.Graph()
        G.add_edges_from(data.edge_index.T.numpy())
        pos = nx.spring_layout(G)
        pos = np.array([pos[i] for i in range(data.num_nodes)])

    comp_labels = data.comp.numpy() if hasattr(data, 'comp') else np.zeros(data.num_nodes)
    n_components = len(np.unique(comp_labels))

    # Color map for components
    colors = plt.cm.tab10(np.linspace(0, 1, n_components))

    # ========== Left plot: Original structure ==========
    ax1.set_title(f"{title}\n(Original + Component Structure)", fontsize=11, fontweight='bold')

    # Plot nodes by component
    for comp_id in range(n_components):
        mask = comp_labels == comp_id
        ax1.scatter(pos[mask, 0], pos[mask, 1],
                    c=[colors[comp_id]], s=node_size,
                    label=f'Component {comp_id}', alpha=0.8, edgecolors='black', linewidth=1)

    # Plot original edges only
    if hasattr(data, 'edge_type'):
        edge_mask = data.edge_type == 0
        edges_to_plot = data.edge_index[:, edge_mask]
    else:
        edges_to_plot = data.edge_index

    for i in range(edges_to_plot.shape[1]):
        src, dst = edges_to_plot[:, i]
        ax1.plot([pos[src, 0], pos[dst, 0]],
                 [pos[src, 1], pos[dst, 1]],
                 'k-', alpha=0.3, linewidth=0.5)

    # ========== Right plot: Hyperedge structure ==========
    ax2.set_title(f"{title}\n(Hyperedge Connections)", fontsize=11, fontweight='bold')

    # Plot nodes by component
    for comp_id in range(n_components):
        mask = comp_labels == comp_id
        ax2.scatter(pos[mask, 0], pos[mask, 1],
                    c=[colors[comp_id]], s=node_size,
                    alpha=0.8, edgecolors='black', linewidth=1)

    # Plot original edges (faint)
    if hasattr(data, 'edge_type'):
        edge_mask = data.edge_type == 0
        edges_orig = data.edge_index[:, edge_mask]
        for i in range(edges_orig.shape[1]):
            src, dst = edges_orig[:, i]
            ax2.plot([pos[src, 0], pos[dst, 0]],
                     [pos[src, 1], pos[dst, 1]],
                     'gray', alpha=0.1, linewidth=0.5)

    # Plot hyperedges
    if hasattr(data, 'hyperedges'):
        hyperedge_colors = plt.cm.Set2(np.linspace(0, 1, len(data.hyperedges)))

        for he_idx, hyperedge in enumerate(data.hyperedges):
            if len(hyperedge) < 2:
                continue

            he_pos = pos[hyperedge]

            # Draw convex hull around hyperedge
            if show_hyperedge_hulls and len(hyperedge) >= 3:
                try:
                    hull = ConvexHull(he_pos)
                    hull_points = he_pos[hull.vertices]
                    poly = Polygon(hull_points, alpha=0.15,
                                   facecolor=hyperedge_colors[he_idx],
                                   edgecolor=hyperedge_colors[he_idx], linewidth=2)
                    ax2.add_patch(poly)
                except:
                    pass  # Skip if hull computation fails

            # Draw edges within hyperedge
            for i in range(len(hyperedge)):
                for j in range(i + 1, len(hyperedge)):
                    v1, v2 = hyperedge[i], hyperedge[j]
                    ax2.plot([pos[v1, 0], pos[v2, 0]],
                             [pos[v1, 1], pos[v2, 1]],
                             color=hyperedge_colors[he_idx],
                             alpha=0.7, linewidth=2, linestyle='--')

            # Mark hyperedge nodes
            ax2.scatter(he_pos[:, 0], he_pos[:, 1],
                        s=node_size * 1.5, facecolors='none',
                        edgecolors=hyperedge_colors[he_idx], linewidth=2)

    # Plot hyperedge connections (if edge_type available)
    elif hasattr(data, 'edge_type'):
        edge_mask = data.edge_type == 1
        edges_hyper = data.edge_index[:, edge_mask]
        for i in range(edges_hyper.shape[1]):
            src, dst = edges_hyper[:, i]
            ax2.plot([pos[src, 0], pos[dst, 0]],
                     [pos[src, 1], pos[dst, 1]],
                     'r--', alpha=0.5, linewidth=1.5)

    # Formatting
    for ax in [ax1, ax2]:
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')

    if show_legend:
        ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # Add info text
    if hasattr(data, 'hypergraph_info'):
        info = data.hypergraph_info
        info_text = f"Algorithm: {info['algorithm']}, s={info['s']}, "
        info_text += f"Components: {info['num_components']}, "
        info_text += f"Bridging edges: {info['num_bridging_edges']}"
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved visualization to {save_path}")
    else:
        plt.show()

    return fig


def analyze_hypergraph_connectivity(data: Data) -> Dict:

    # Create NetworkX graph for analysis
    G = nx.Graph()
    edges = data.edge_index.T.numpy()
    G.add_edges_from(edges)

    # Basic connectivity
    is_connected = nx.is_connected(G)
    n_components = nx.number_connected_components(G)

    # Component analysis
    comp_labels = data.comp.numpy() if hasattr(data, 'comp') else np.zeros(data.num_nodes)
    n_original_components = len(np.unique(comp_labels))

    # Edge type analysis
    if hasattr(data, 'edge_type'):
        n_original_edges = (data.edge_type == 0).sum().item() // 2  # Undirected
        n_hyperedge_connections = (data.edge_type == 1).sum().item() // 2
    else:
        n_original_edges = data.edge_index.shape[1] // 2
        n_hyperedge_connections = 0

    # Hyperedge analysis
    if hasattr(data, 'hyperedges'):
        hyperedge_sizes = [len(he) for he in data.hyperedges]
        avg_hyperedge_size = np.mean(hyperedge_sizes) if hyperedge_sizes else 0
        max_hyperedge_size = max(hyperedge_sizes) if hyperedge_sizes else 0
    else:
        hyperedge_sizes = []
        avg_hyperedge_size = 0
        max_hyperedge_size = 0

    # Degree analysis
    degrees = dict(G.degree())
    avg_degree = np.mean(list(degrees.values()))
    max_degree = max(degrees.values()) if degrees else 0

    # Shortest path analysis (sample if graph is large)
    if G.number_of_nodes() <= 100 and is_connected:
        avg_path_length = nx.average_shortest_path_length(G)
        diameter = nx.diameter(G)
    else:
        avg_path_length = -1
        diameter = -1

    return {
        'is_connected': is_connected,
        'n_connected_components': n_components,
        'n_original_components': n_original_components,
        'n_nodes': data.num_nodes,
        'n_edges': data.num_edges // 2,  # Undirected
        'n_original_edges': n_original_edges,
        'n_hyperedge_connections': n_hyperedge_connections,
        'n_hyperedges': len(data.hyperedges) if hasattr(data, 'hyperedges') else 0,
        'hyperedge_sizes': hyperedge_sizes,
        'avg_hyperedge_size': avg_hyperedge_size,
        'max_hyperedge_size': max_hyperedge_size,
        'avg_degree': avg_degree,
        'max_degree': max_degree,
        'avg_shortest_path': avg_path_length,
        'diameter': diameter,
    }


def load_and_process_results(json_file, filter=False):
    """Load results from JSON file and process into aggregated DataFrame"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    raw_data = data['raw_data']
    parameters = data['parameters']

    # Process the data into a structured format
    processed_data = []

    for algorithm, epochs_data in raw_data.items():
        for epoch_str, epoch_data in epochs_data.items():
            epoch = int(epoch_str)
            runs_data = epoch_data['runs']

            # Extract values from all runs
            first_terms = [run['first_term'] for run in runs_data]
            second_terms = [run['second_term'] for run in runs_data]
            graph_losses = [run['graph_loss'] for run in runs_data]
            total_losses = [run['total_loss'] for run in runs_data]

            # Filter outliers using IQR method for each metric
            def filter_outliers(values):
                """Remove outliers using IQR method"""

                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                filtered = [v for v in values if lower_bound <= v <= upper_bound]

                return filtered

            # Apply outlier filtering
            if filter:
                first_terms = filter_outliers(first_terms)
                second_terms = filter_outliers(second_terms)
                graph_losses = filter_outliers(graph_losses)
                total_losses = filter_outliers(total_losses)

            # Calculate statistics
            processed_data.append({
                'Algorithm': algorithm,
                'Epoch': epoch,
                'First Term Mean': np.mean(first_terms),
                'First Term Std': np.std(first_terms),
                'Second Term Mean': np.mean(second_terms),
                'Second Term Std': np.std(second_terms),
                'Graph Loss Mean': np.mean(graph_losses),
                'Graph Loss Std': np.std(graph_losses),
                'Total Loss Mean': np.mean(total_losses),
                'Total Loss Std': np.std(total_losses),
                'Num Runs': len(runs_data),
                'Num Runs After Filtering': len(total_losses)
            })

    return pd.DataFrame(processed_data), parameters


def plot_comparison_results(df, parameters, save_dir="Metrics"):
    """Create comprehensive plots comparing different algorithms"""

    os.makedirs(save_dir, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Get unique algorithms and epochs
    algorithms = df['Algorithm'].unique()
    epochs = sorted(df['Epoch'].unique())

    # Create figure with subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Graph Algorithm Metrics - Loss Components Over Epochs\n'
                 f'(Learning Rate: {parameters["learning_rate"]}, Runs: {parameters["num_runs"]})',
                 fontsize=16)

    metrics = [
        ('First Term Mean', 'First Term Std', 'First Term Loss (KL Divergence)'),
        ('Second Term Mean', 'Second Term Std', 'Second Term Loss (MSE)'),
        ('Graph Loss Mean', 'Graph Loss Std', 'Graph Sparsity Loss'),
        ('Total Loss Mean', 'Total Loss Std', 'Total Loss')
    ]

    for idx, (metric, std_metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        for algo in algorithms:
            algo_data = df[df['Algorithm'] == algo].sort_values('Epoch')
            values = algo_data[metric].values
            stds = algo_data[std_metric].values

            # Plot mean with error bars
            ax.errorbar(epochs, values, yerr=stds, label=algo, marker='o',
                        capsize=5, capthick=2, linewidth=2, markersize=8)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss Value (Log Scale)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_components_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Create individual plots for better detail
    for metric, std_metric, title in metrics:
        plt.figure(figsize=(10, 6))

        for algo in algorithms:
            algo_data = df[df['Algorithm'] == algo].sort_values('Epoch')
            values = algo_data[metric].values
            stds = algo_data[std_metric].values

            plt.errorbar(epochs, values, yerr=stds, label=algo, marker='o',
                         capsize=5, capthick=2, linewidth=2, markersize=8)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.title(f'{title} - Detailed Metrics', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        filename = title.lower().replace(' ', '_').replace('(', '').replace(')', '') + '_detailed.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()

    # Create heatmap showing final epoch performance
    final_epoch = max(epochs)
    final_data = df[df['Epoch'] == final_epoch]

    # Prepare data for heatmap
    heatmap_data = []
    for algo in algorithms:
        algo_final = final_data[final_data['Algorithm'] == algo]
        if not algo_final.empty:
            heatmap_data.append([
                algo_final['First Term Mean'].values[0],
                algo_final['Second Term Mean'].values[0],
                algo_final['Graph Loss Mean'].values[0],
                algo_final['Total Loss Mean'].values[0]
            ])

    # Create heatmap
    plt.figure(figsize=(10, 6))
    heatmap_df = pd.DataFrame(heatmap_data,
                              index=algorithms,
                              columns=['First Term', 'Second Term', 'Graph Loss', 'Total Loss'])

    # Normalize each column for better visualization
    heatmap_normalized = heatmap_df.div(heatmap_df.max(axis=0), axis=1)

    sns.heatmap(heatmap_normalized, annot=heatmap_df.values, fmt='.6f',
                cmap='RdYlGn_r', cbar_kws={'label': 'Normalized Loss'})
    plt.title(f'Final Performance Metrics (Epoch {final_epoch})', fontsize=14)
    plt.xlabel('Loss Component', fontsize=12)
    plt.ylabel('Algorithm', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_performance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Create convergence speed plot
    plt.figure(figsize=(10, 6))

    for algo in algorithms:
        algo_data = df[df['Algorithm'] == algo].sort_values('Epoch')
        total_losses = algo_data['Total Loss Mean'].values

        # Calculate relative improvement from epoch 1
        if len(total_losses) > 0:
            relative_improvement = (total_losses[0] - total_losses) / total_losses[0] * 100
            plt.plot(epochs[:len(relative_improvement)], relative_improvement,
                     label=algo, marker='o', linewidth=2, markersize=8)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Relative Improvement (%)', fontsize=12)
    plt.title('Convergence Speed Metrics', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'convergence_speed.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Create stability plot (standard deviation over epochs)
    plt.figure(figsize=(10, 6))

    for algo in algorithms:
        algo_data = df[df['Algorithm'] == algo].sort_values('Epoch')
        total_stds = algo_data['Total Loss Std'].values

        plt.plot(epochs[:len(total_stds)], total_stds,
                 label=algo, marker='o', linewidth=2, markersize=8)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Standard Deviation', fontsize=12)
    plt.title('Training Stability Metrics (Lower is Better)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(os.path.join(save_dir, 'training_stability.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for algo in algorithms:
        algo_data = df[df['Algorithm'] == algo]
        final_algo = final_data[final_data['Algorithm'] == algo]

        if not final_algo.empty:
            print(f"\n{algo}:")
            print(
                f"  Final Total Loss: {final_algo['Total Loss Mean'].values[0]:.6f} ± {final_algo['Total Loss Std'].values[0]:.6f}")
            print(f"  Best Epoch: {algo_data.loc[algo_data['Total Loss Mean'].idxmin(), 'Epoch']}")
            print(f"  Best Total Loss: {algo_data['Total Loss Mean'].min():.6f}")

            # Calculate convergence rate
            total_losses = algo_data['Total Loss Mean'].values
            if len(total_losses) > 0:
                improvement = (total_losses[0] - total_losses[-1]) / total_losses[0] * 100
                print(f"  Total Improvement: {improvement:.2f}%")

            # Report outlier filtering impact
            total_runs = algo_data['Num Runs'].sum()
            filtered_runs = algo_data['Num Runs After Filtering'].sum()
            print(f"  Outliers Filtered: {total_runs - filtered_runs}/{total_runs} runs")

def create_performance_ranking(df, save_dir="Metrics"):
    """Create a bar chart showing algorithm ranking by final performance"""

    final_epoch = df['Epoch'].max()
    final_data = df[df['Epoch'] == final_epoch].copy()

    # Sort by total loss
    final_data = final_data.sort_values('Total Loss Mean')

    plt.figure(figsize=(10, 6))

    # Create bar chart
    bars = plt.bar(range(len(final_data)), final_data['Total Loss Mean'],
                   yerr=final_data['Total Loss Std'], capsize=5)

    # Color bars by performance (best = green, worst = red)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.xticks(range(len(final_data)), final_data['Algorithm'], rotation=45, ha='right')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Final Total Loss', fontsize=12)
    plt.title(f'Algorithm Performance Ranking at Epoch {final_epoch}', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_ranking.png'), dpi=300, bbox_inches='tight')
    plt.show()


def permutation_test(group1, group2, n_permutations=10000, alternative='two-sided'):
    """
    Perform a permutation test to compare two groups.

    Parameters:
    -----------
    group1, group2 : array-like
        The two groups to compare
    n_permutations : int
        Number of permutations to perform
    alternative : str
        'two-sided', 'less', or 'greater'

    Returns:
    --------
    observed_diff : float
        The observed difference in means (group1 - group2)
    p_value : float
        The p-value from the permutation test
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    # Calculate observed difference in means
    observed_diff = np.mean(group1) - np.mean(group2)

    # Combine all observations
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    n_total = len(combined)

    # Generate permutations
    permuted_diffs = []
    np.random.seed(42)  # For reproducibility

    for _ in range(n_permutations):
        # Shuffle the combined data
        np.random.shuffle(combined)

        # Split into two groups of original sizes
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]

        # Calculate difference for this permutation
        perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
        permuted_diffs.append(perm_diff)

    permuted_diffs = np.array(permuted_diffs)

    # Calculate p-value based on alternative hypothesis
    if alternative == 'two-sided':
        # Count how many permuted differences are as extreme or more extreme
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
    elif alternative == 'less':
        # Test if group1 < group2
        p_value = np.mean(permuted_diffs <= observed_diff)
    elif alternative == 'greater':
        # Test if group1 > group2
        p_value = np.mean(permuted_diffs >= observed_diff)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    return observed_diff, p_value, permuted_diffs


def filter_outliers_iqr(values, iqr_multiplier=1.5):
    """
    Remove outliers using IQR (Interquartile Range) method.

    Parameters:
    -----------
    values : array-like
        The values to filter
    iqr_multiplier : float
        Multiplier for IQR to determine outlier bounds (default: 1.5)
        Common values: 1.5 (standard), 1.0 (aggressive), 2.0 (conservative)

    Returns:
    --------
    filtered_values : list
        Values with outliers removed
    outlier_indices : list
        Indices of outliers in original array
    """
    if len(values) < 3:  # Not enough data to filter outliers
        return values, []

    values_array = np.array(values)
    q1 = np.percentile(values_array, 25)
    q3 = np.percentile(values_array, 75)
    iqr = q3 - q1

    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    # Find outlier indices
    outlier_mask = (values_array < lower_bound) | (values_array > upper_bound)
    outlier_indices = np.where(outlier_mask)[0].tolist()

    # Filter values
    filtered_values = values_array[~outlier_mask].tolist()

    # If we filter out too many values (more than 50%), return original
    if len(filtered_values) < len(values) * 0.5:
        return values, []

    return filtered_values, outlier_indices


def perform_statistical_comparison(raw_data, df, epoch=30, save_dir="Metrics",
                                   n_permutations=10000, filter_outliers=False,
                                   iqr_multiplier=1.5):
    """
    Perform permutation tests between kNN with Gabriel Pruning and its closest competitor at specified epoch.

    Parameters:
    -----------
    raw_data : dict
        Raw experimental data
    df : DataFrame
        Processed results dataframe
    epoch : int
        Epoch number to analyze
    save_dir : str
        Directory to save plots
    n_permutations : int
        Number of permutations for the test
    filter_outliers : bool
        Whether to filter outliers using IQR method
    iqr_multiplier : float
        Multiplier for IQR to determine outlier bounds (default: 1.5)
    """

    print("\n" + "=" * 80)
    print(f"STATISTICAL COMPARISON AT EPOCH {epoch}")
    print("=" * 80)

    if filter_outliers:
        print(f"Outlier filtering: ENABLED (IQR multiplier: {iqr_multiplier})")
    else:
        print("Outlier filtering: DISABLED")

    # Check if kNN with Gabriel Pruning exists and epoch data is available
    if 'kNN with Gabriel Pruning' not in raw_data or str(epoch) not in raw_data['kNN with Gabriel Pruning']:
        print(f"kNN with Gabriel Pruning data not found for epoch {epoch}")
        return

    # Get kNN with Gabriel Pruning's performance at specified epoch
    knn_gabriel_pruning_epoch_data = df[(df['Algorithm'] == 'kNN with Gabriel Pruning') & (df['Epoch'] == epoch)]
    if knn_gabriel_pruning_epoch_data.empty:
        print(f"No kNN with Gabriel Pruning data found for epoch {epoch}")
        return

    knn_gabriel_pruning_total_loss = knn_gabriel_pruning_epoch_data['Total Loss Mean'].values[0]

    # Find closest competitor based on total loss at specified epoch
    other_algos_epoch = df[(df['Algorithm'] != 'kNN with Gabriel Pruning') & (df['Epoch'] == epoch)].copy()
    if other_algos_epoch.empty:
        print("No other algorithms found for Metrics")
        return

    other_algos_epoch['Diff_from_kNN_with_Gabriel_Pruning'] = abs(other_algos_epoch['Total Loss Mean'] - knn_gabriel_pruning_total_loss)
    closest_competitor = other_algos_epoch.loc[other_algos_epoch['Diff_from_kNN_with_Gabriel_Pruning'].idxmin(), 'Algorithm']

    print(f"\nkNN_with_Gabriel_Pruning Total Loss at Epoch {epoch}: {knn_gabriel_pruning_total_loss:.6f}")
    print(f"Closest Competitor: {closest_competitor}")
    print(f"{closest_competitor} Total Loss at Epoch {epoch}: "
          f"{other_algos_epoch[other_algos_epoch['Algorithm'] == closest_competitor]['Total Loss Mean'].values[0]:.6f}")

    # Get raw data for both algorithms at specified epoch
    knn_gabriel_pruning_runs = raw_data['kNN with Gabriel Pruning'][str(epoch)]['runs']
    competitor_runs = raw_data[closest_competitor][str(epoch)]['runs']

    # Extract values for each metric
    metrics = [
        ('first_term', 'First Term (KL Divergence)'),
        ('second_term', 'Second Term (MSE)'),
        ('graph_loss', 'Graph Loss (Sparsity)'),
        ('total_loss', 'Total Loss')
    ]

    print(f"\n{'=' * 60}")
    print(f"Permutation Tests: kNN with Gabriel Pruning vs {closest_competitor}")
    print(f"{'=' * 60}")
    print(f"Number of runs: kNN with Gabriel Pruning={len(knn_gabriel_pruning_runs)}, {closest_competitor}={len(competitor_runs)}")
    print(f"Number of permutations: {n_permutations}")

    if len(knn_gabriel_pruning_runs) < 3 or len(competitor_runs) < 3:
        print("\nWARNING: Sample size is very small (< 3). Results should be interpreted with caution.")
        print("Consider collecting more runs for more reliable statistical inference.")

    if len(knn_gabriel_pruning_runs) < 2 or len(competitor_runs) < 2:
        print("\nERROR: Cannot perform Metrics with fewer than 2 samples per group.")
        print("Please collect more runs before performing statistical Metrics.")
        return

    results_summary = []
    outlier_summary = []  # Track outlier removal

    # Store permutation distributions for visualization
    all_permutation_results = {}

    for metric_key, metric_name in metrics:
        knn_gabriel_pruning_values_raw = [run[metric_key] for run in knn_gabriel_pruning_runs]
        competitor_values_raw = [run[metric_key] for run in competitor_runs]

        # Apply outlier filtering if requested
        if filter_outliers:
            knn_gabriel_pruning_values, knn_gabriel_pruning_outlier_idx = filter_outliers_iqr(knn_gabriel_pruning_values_raw, iqr_multiplier)
            competitor_values, comp_outlier_idx = filter_outliers_iqr(competitor_values_raw, iqr_multiplier)

            outlier_summary.append({
                'Metric': metric_name,
                'kNN_with_Gabriel_Pruning_outliers': len(knn_gabriel_pruning_outlier_idx),
                'Competitor_outliers': len(comp_outlier_idx),
                'kNN_with_Gabriel_Pruning_remaining': len(knn_gabriel_pruning_values),
                'Competitor_remaining': len(competitor_values)
            })

            # Keep track of which values are outliers for visualization
            knn_gabriel_pruning_outliers = [knn_gabriel_pruning_values_raw[i] for i in knn_gabriel_pruning_outlier_idx]
            comp_outliers = [competitor_values_raw[i] for i in comp_outlier_idx]
        else:
            knn_gabriel_pruning_values = knn_gabriel_pruning_values_raw
            competitor_values = competitor_values_raw
            knn_gabriel_pruning_outliers = []
            comp_outliers = []

        # Check if we have enough data after filtering
        if filter_outliers and (len(knn_gabriel_pruning_values) < 2 or len(competitor_values) < 2):
            print(f"\nERROR: Too few samples remain after outlier filtering for {metric_name}")
            print(f"       kNN with Gabriel Pruning: {len(knn_gabriel_pruning_values)}, {closest_competitor}: {len(competitor_values)}")
            print("       Skipping this metric...")
            continue

        # Perform permutation test
        # Since lower loss is better, we use 'less' to test if kNN with Gabriel Pruning < competitor
        observed_diff, p_value, permuted_diffs = permutation_test(
            knn_gabriel_pruning_values, competitor_values, n_permutations=n_permutations, alternative='two-sided'
        )

        all_permutation_results[metric_key] = {
            'observed_diff': observed_diff,
            'permuted_diffs': permuted_diffs,
            'p_value': p_value,
            'knn_gabriel_pruning_outliers': knn_gabriel_pruning_outliers,
            'comp_outliers': comp_outliers
        }

        # Calculate descriptive statistics
        knn_gabriel_pruning_mean = np.mean(knn_gabriel_pruning_values)
        knn_gabriel_pruning_std = np.std(knn_gabriel_pruning_values, ddof=1)
        knn_gabriel_pruning_median = np.median(knn_gabriel_pruning_values)
        comp_mean = np.mean(competitor_values)
        comp_std = np.std(competitor_values, ddof=1)
        comp_median = np.median(competitor_values)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(knn_gabriel_pruning_values) - 1) * knn_gabriel_pruning_std ** 2 +
                              (len(competitor_values) - 1) * comp_std ** 2) /
                             (len(knn_gabriel_pruning_values) + len(competitor_values) - 2))

        cohen_d = (knn_gabriel_pruning_mean - comp_mean) / pooled_std if pooled_std > 0 else 0

        print(f"\n{metric_name}:")
        if filter_outliers and outlier_summary:
            metric_outliers = outlier_summary[-1]  # Last added entry
            print(f"  Outliers removed: kNN with Gabriel Pruning={metric_outliers['kNN_with_Gabriel_Pruning_outliers']}, "
                  f"{closest_competitor}={metric_outliers['Competitor_outliers']}")
        print(f"  kNN with Gabriel Pruning:     mean={knn_gabriel_pruning_mean:.6f}, std={knn_gabriel_pruning_std:.6f}, median={knn_gabriel_pruning_median:.6f}")
        print(f"  {closest_competitor}: mean={comp_mean:.6f}, std={comp_std:.6f}, median={comp_median:.6f}")
        print(f"  Observed difference (kNN with Gabriel Pruning - {closest_competitor}): {observed_diff:.6f}")
        print(f"  Permutation test p-value: {p_value:.4f}")
        print(f"  Cohen's d: {cohen_d:.4f}")

        # Interpret results
        if p_value < 0.05:
            better = "kNN with Gabriel Pruning" if observed_diff < 0 else closest_competitor
            print(f"  Result: Statistically significant difference (p < 0.05)")
            print(f"          {better} performs significantly better")
        else:
            print(f"  Result: No statistically significant difference (p >= 0.05)")
            print(f"          The observed difference could be due to chance")

        results_summary.append({
            'Metric': metric_name,
            'Observed Diff': observed_diff,
            'p-value': p_value,
            'Significant': p_value < 0.05,
            'Cohen\'s d': cohen_d
        })

    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))

    # Create a 4x3 grid: 4 metrics × (boxplot + histogram + QQ plot)
    for idx, (metric_key, metric_name) in enumerate(metrics):
        if filter_outliers:
            knn_gabriel_pruning_values_raw = [run[metric_key] for run in knn_gabriel_pruning_runs]
            competitor_values_raw = [run[metric_key] for run in competitor_runs]
            knn_gabriel_pruning_values, _ = filter_outliers_iqr(knn_gabriel_pruning_values_raw, iqr_multiplier)
            competitor_values, _ = filter_outliers_iqr(competitor_values_raw, iqr_multiplier)
        else:
            knn_gabriel_pruning_values = [run[metric_key] for run in knn_gabriel_pruning_runs]
            competitor_values = [run[metric_key] for run in competitor_runs]

        perm_results = all_permutation_results[metric_key]

        # Boxplot
        ax1 = plt.subplot(4, 3, idx * 3 + 1)
        bp = ax1.boxplot([knn_gabriel_pruning_values, competitor_values],
                         tick_labels=['kNN with Gabriel Pruning', closest_competitor],
                         patch_artist=True, showmeans=True,
                         meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightgreen')

        # Add individual points
        x1 = np.random.normal(1, 0.04, len(knn_gabriel_pruning_values))
        x2 = np.random.normal(2, 0.04, len(competitor_values))
        ax1.scatter(x1, knn_gabriel_pruning_values, alpha=0.5, s=30, color='blue', label='Data points')
        ax1.scatter(x2, competitor_values, alpha=0.5, s=30, color='green')

        # Add outliers as red X marks if filtering is enabled
        if filter_outliers and (perm_results['knn_gabriel_pruning_outliers'] or perm_results['comp_outliers']):
            if perm_results['knn_gabriel_pruning_outliers']:
                x1_out = np.random.normal(1, 0.04, len(perm_results['knn_gabriel_pruning_outliers']))
                ax1.scatter(x1_out, perm_results['knn_gabriel_pruning_outliers'], alpha=0.8, s=50,
                            color='red', marker='x', linewidth=2, label='Outliers (removed)')
            if perm_results['comp_outliers']:
                x2_out = np.random.normal(2, 0.04, len(perm_results['comp_outliers']))
                ax1.scatter(x2_out, perm_results['comp_outliers'], alpha=0.8, s=50,
                            color='red', marker='x', linewidth=2)
            ax1.legend(fontsize=8, loc='upper right')

        ax1.set_ylabel('Loss Value', fontsize=10)
        ax1.set_title(f'{metric_name}\nBoxplot Metrics', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Permutation distribution
        ax2 = plt.subplot(4, 3, idx * 3 + 2)
        ax2.hist(perm_results['permuted_diffs'], bins=50, alpha=0.7,
                 color='gray', edgecolor='black', density=True)
        ax2.axvline(perm_results['observed_diff'], color='red', linestyle='--',
                    linewidth=2, label=f'Observed diff: {perm_results["observed_diff"]:.6f}')
        ax2.axvline(0, color='black', linestyle='-', alpha=0.5)

        # Add shaded area for p-value
        if perm_results['p_value'] < 0.05:
            ax2.set_facecolor('#ffe6e6')  # Light red background for significant results

        ax2.set_xlabel('Difference in Means', fontsize=10)
        ax2.set_ylabel('Density', fontsize=10)
        ax2.set_title(f'Permutation Distribution\np-value: {perm_results["p_value"]:.4f}', fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Q-Q plot to check normality
        ax3 = plt.subplot(4, 3, idx * 3 + 3)

        # Combine data for Q-Q plot
        all_values = knn_gabriel_pruning_values + competitor_values
        stats.probplot(all_values, dist="norm", plot=ax3)
        ax3.set_title(f'Q-Q Plot (Normality Check)\nCombined Data', fontsize=10)
        ax3.grid(True, alpha=0.3)

    filter_text = f" (Outliers Filtered, IQR×{iqr_multiplier})" if filter_outliers else ""
    plt.suptitle(f'Statistical Metrics: kNN with Gabriel Pruning vs {closest_competitor} at Epoch {epoch}\n'
                 f'Permutation Test with {n_permutations} permutations{filter_text}', fontsize=14)
    plt.tight_layout()
    filter_suffix = "_filtered" if filter_outliers else ""
    plt.savefig(os.path.join(save_dir, f'permutation_test_comparison_epoch{epoch}{filter_suffix}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY OF PERMUTATION TESTS")
    print(f"{'=' * 80}")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

    print("\nEffect Size Interpretation (Cohen's d):")
    print("  |d| < 0.2: Negligible")
    print("  0.2 ≤ |d| < 0.5: Small")
    print("  0.5 ≤ |d| < 0.8: Medium")
    print("  |d| ≥ 0.8: Large")

    print("\nPermutation Test Interpretation:")
    print("  - The p-value represents the probability of observing a difference")
    print("    as extreme as the one observed, assuming no true difference exists.")
    print("  - Unlike t-tests, permutation tests make no assumptions about the")
    print("    underlying distribution of the data.")
    print("  - Results are based on empirical distributions from random permutations.")

    # Print outlier summary if filtering was applied
    if filter_outliers:
        print(f"\n{'=' * 80}")
        print("OUTLIER FILTERING SUMMARY")
        print(f"{'=' * 80}")
        outlier_df = pd.DataFrame(outlier_summary)
        print(outlier_df.to_string(index=False))

        total_knn_gabriel_pruning_removed = sum(row['kNN_with_Gabriel_Pruning_outliers'] for row in outlier_summary)
        total_comp_removed = sum(row['Competitor_outliers'] for row in outlier_summary)
        print(f"\nTotal outliers removed: kNN with Gabriel Pruning={total_knn_gabriel_pruning_removed}, {closest_competitor}={total_comp_removed}")
        print(f"IQR multiplier used: {iqr_multiplier} (lower = more aggressive filtering)")