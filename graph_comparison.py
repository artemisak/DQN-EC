import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from datetime import datetime
import json
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
from pettingzoo.mpe import simple_speaker_listener_v4

from graphs_wrapper import (
    create_delaunay_graph,
    create_beta_skeleton_graph,
    create_gabriel_graph,
    create_knn_gabriel_pruning_graph,
    create_knn_graph,
    create_full_connected_graph
)


# Gray encoding
GRAY_FORWARD_MAPPING = {
    1: [0, 0, 0, 0], # agent_type
    2: [0, 0, 0, 1], # self_velocity_x
    3: [0, 0, 1, 1], # self_velocity_y
    4: [0, 0, 1, 0], # landmark_1_rel_x
    5: [0, 1, 1, 0], # landmark_1_rel_y
    6: [0, 1, 1, 1], # landmark_2_rel_x
    7: [0, 1, 0, 1], # landmark_2_rel_y
    8: [0, 1, 0, 0], # landmark_3_rel_x
    9: [1, 1, 0, 0], # landmark_3_rel_y
    10: [1, 1, 0, 1], # landmark_1_is_target
    11: [1, 1, 1, 1], # landmark_2_is_target
    12: [1, 1, 1, 0] # landmark_3_is_target
}

GRAY_INVERTED_MAPPING = {
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
    (1, 1, 1, 1): 11,
    (1, 1, 1, 0): 12
}

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Generate sample data
def generate_sample_data(num_samples=1024, batch_size=64):
    """Sample the data from the environment with a random policy"""
    
    env = simple_speaker_listener_v4.parallel_env(max_cycles=num_samples)
    env.reset()
    observations = []
    
    def create_observation_pair(message):
        """Create both observation variants for a message"""

        # First observation (agent_type=0)
        obs1 = torch.zeros(12, 5)
        obs1[:, :4] = torch.tensor(list(GRAY_FORWARD_MAPPING.values()))  # labels
        obs1[0, 4] = 0 # agent_type
        obs1[1:9, 4] = torch.tensor(message[:8])  # velocity and landmarks
        obs1[9:, 4] = -1 # masked out

        # First observation (agent_type=0)
        obs2 = torch.zeros(12, 5)
        obs2[:, :4] = torch.tensor(list(GRAY_FORWARD_MAPPING.values()))  # labels
        obs2[0, 4] = 1 # agent_type
        obs2[1:9, 4] = -1  # masked out
        obs2[9:, 4] = torch.tensor(message[8:])  # target flags

        perm = torch.randperm(len(GRAY_FORWARD_MAPPING))
        obs1 = obs1[perm]
        obs2 =  obs2[perm]

        return [obs1, obs2]

    while env.agents and len(observations) < num_samples:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, *_ = env.step(actions)
        observations.extend(create_observation_pair(obs['listener_0']))
    
    env.close()

    observations = torch.stack(observations)
    perm = torch.randperm(len(observations))
    observations = observations[perm]

    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(observations.to(device)),
        batch_size=batch_size,
        shuffle=True
    )


class GraphAutoEncoderComparison(nn.Module):
    """Modified GraphAutoEncoder that accepts graph creation function as parameter"""

    def __init__(self, input_dim=5, output_dim=3, hidden_dim=128, graph_fn=None):
        super(GraphAutoEncoderComparison, self).__init__()

        self.graph_fn = graph_fn if graph_fn is not None else create_gabriel_graph

        # Encoder MLP
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.encoder.apply(self.kaiming_init)

        # Shared GAT layers
        self.gcn1 = GATv2Conv(in_channels=1, out_channels=hidden_dim, edge_dim=1)
        self.gcn2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=1)

        # Label prediction
        self.gcn3 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.label_head = nn.Linear(hidden_dim, 4)

        # Value prediction
        self.gcn4 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Skip connection layer
        self.skip_connection = nn.Linear(output_dim, hidden_dim)

        # Alpha for the skip connection
        self.alpha = 0.1

    def kaiming_init(self, m):
        """Helper method for Kaiming initialization"""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def forward(self, batch):
        reconstructed_labels = []
        reconstructed_values = []
        latent_list = []
        edge_index_list = []
        edge_attr_list = []

        for _, obs in enumerate(batch):
            # Encode each vector to latent vector
            latent = self.encoder(obs)

            # Create a graph using the specified function
            edge_index, edge_attr = self.graph_fn(latent[:, :2])

            # Create PyTorch Geometric Data object
            graph = Data(x=latent[:, 2].reshape(-1, 1), edge_index=edge_index, edge_attr=edge_attr)

            x1 = F.relu(self.gcn1(graph.x, graph.edge_index, edge_attr=graph.edge_attr))
            x2 = F.relu(self.gcn2(x1, graph.edge_index, edge_attr=graph.edge_attr))

            # Predict the label
            x3 = F.relu(self.gcn3(x2, graph.edge_index) + self.alpha * self.skip_connection(latent))
            logits = self.label_head(x3)

            # Predict the value
            x4 = F.relu(self.gcn4(x2, graph.edge_index) + self.alpha * self.skip_connection(latent))
            values = self.value_head(x4)

            reconstructed_labels.append(logits)
            reconstructed_values.append(values)
            latent_list.append(latent)
            edge_index_list.append(edge_index)
            edge_attr_list.append(edge_attr)

        return (batch[:, :, :4], batch[:, :, 4].reshape(64, 12, 1),
                torch.stack(reconstructed_labels), torch.stack(reconstructed_values),
                torch.stack(latent_list), edge_index_list, edge_attr_list)


def reconstruction_loss_components(true_distribution, predicted_logits, true_values, predicted_values, epoch,
                                   total_epochs):
    """
    Calculate reconstruction loss components separately
    """
    labels_kl = F.kl_div(F.log_softmax(predicted_logits, dim=-1), F.softmax(true_distribution, dim=-1),
                         reduction='batchmean')
    values_mse = F.l1_loss(predicted_values, true_values)

    progress = epoch / total_epochs
    weight1 = 1.0 - 0.9 * progress
    weight2 = 0.1 + 0.9 * progress

    first_term = weight1 * labels_kl
    second_term = weight2 * values_mse
    total = first_term + second_term

    return first_term, second_term, total, labels_kl, values_mse


def l1_loss(edge_attr_list):
    """L1 loss for the edges on the graph"""
    return torch.norm(torch.cat(edge_attr_list), p=1)


def visualize_latent_graph(latent_points, edge_index, edge_attr, algo_name, save_path):
    """
    Visualize the latent space graph and save as PNG
    
    Args:
        latent_points: 2D points in latent space (N x 2)
        edge_index: Edge indices (2 x E)
        edge_attr: Edge attributes/weights (E,)
        algo_name: Name of the algorithm
        save_path: Path to save the PNG
    """
    # Convert to numpy for plotting
    points = latent_points.detach().cpu().numpy()
    edges = edge_index.detach().cpu().numpy()
    weights = edge_attr.detach().cpu().numpy() if edge_attr is not None else None
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Plot edges
    if edges.shape[1] > 0:
        # Create line segments for edges
        lines = []
        for i in range(edges.shape[1]):
            start_idx = edges[0, i]
            end_idx = edges[1, i]
            lines.append([points[start_idx], points[end_idx]])
        
        # Create LineCollection
        if weights is not None:
            # Normalize weights for coloring
            weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
            lc = LineCollection(lines, cmap='viridis', linewidths=1.5)
            lc.set_array(weights_norm.flatten())
            ax.add_collection(lc)
            
            # Add colorbar
            cbar = plt.colorbar(lc, ax=ax)
            cbar.set_label('Edge Weight (normalized)', fontsize=12)
        else:
            lc = LineCollection(lines, colors='gray', linewidths=1.0, alpha=0.6)
            ax.add_collection(lc)
    
    # Plot nodes
    scatter = ax.scatter(points[:, 0], points[:, 1], 
                        c='red', s=100, zorder=5, 
                        edgecolors='black', linewidths=1.5)
    
    # Set axis properties
    ax.set_xlabel('Latent Dimension 1', fontsize=14)
    ax.set_ylabel('Latent Dimension 2', fontsize=14)
    ax.set_title(f'{algo_name}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits with some padding
    margin = 0.1
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    ax.set_xlim(points[:, 0].min() - margin * x_range, points[:, 0].max() + margin * x_range)
    ax.set_ylim(points[:, 1].min() - margin * y_range, points[:, 1].max() + margin * y_range)
    
    # Add statistics
    num_nodes = points.shape[0]
    num_edges = edges.shape[1]
    avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
    
    stats_text = f'Nodes: {num_nodes}\nEdges: {num_edges}\nAvg Degree: {avg_degree:.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def train_and_track_metrics(model, dataloader, epochs, lr, track_epochs):
    """Train model and track metrics at specific epochs"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    metrics = {epoch: {} for epoch in track_epochs}
    last_epoch_graph_data = None

    for epoch in range(epochs):
        model.train()
        epoch_metrics = {
            'first_term': [],
            'second_term': [],
            'graph_loss': [],
            'total_loss': []
        }

        for batch_idx, (batch,) in enumerate(dataloader):
            # Forward pass
            true_distribution, true_values, predicted_logits, predicted_values, latent_batch, edge_index_list, edge_attr_list = model(
                batch)

            # Calculate loss components
            first_term, second_term, recon_loss, _, _ = reconstruction_loss_components(
                true_distribution, predicted_logits, true_values, predicted_values, epoch, epochs
            )

            graph_loss_val = 1e-6 * l1_loss(edge_attr_list)
            total_loss = recon_loss + graph_loss_val

            # Store metrics
            epoch_metrics['first_term'].append(first_term.item())
            epoch_metrics['second_term'].append(second_term.item())
            epoch_metrics['graph_loss'].append(graph_loss_val.item())
            epoch_metrics['total_loss'].append(total_loss.item())

            # Save graph data from last epoch, first batch
            if epoch == epochs - 1 and batch_idx == 0:
                # Use the first sample in the batch
                last_epoch_graph_data = {
                    'latent': latent_batch[0],  # First sample
                    'edge_index': edge_index_list[0],
                    'edge_attr': edge_attr_list[0]
                }

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Calculate average metrics for the epoch
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}

        # Store metrics if this is a tracked epoch
        if epoch + 1 in track_epochs:
            metrics[epoch + 1] = avg_metrics
            print(f"Epoch {epoch + 1}: Total Loss = {avg_metrics['total_loss']:.6f}")

        # Update learning rate
        scheduler.step(avg_metrics['total_loss'])

    return metrics, last_epoch_graph_data


def run_comparison():
    """Run comparison of different graph algorithms"""

    # Define algorithms to test
    algorithms = {
        'kNN with Gabriel Pruning': create_knn_gabriel_pruning_graph,
        'Beta Skeleton (β=1.7)': lambda points: create_beta_skeleton_graph(points, beta=1.7),
        'Beta Skeleton (β=1.0)': create_gabriel_graph,
        'Fully Connected': create_full_connected_graph,
        'Delaunay': create_delaunay_graph,
        'kNN': create_knn_graph,
    }

    # Parameters
    num_runs = 100
    epochs = 30
    lr = 0.0025
    track_epochs = [1, 5, 10, 15, 20, 25, 30]

    # Create visualization directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_dir = f'graph_visualizations_{timestamp}'
    os.makedirs(viz_dir, exist_ok=True)

    # Results storage
    all_results = []  # For averaged results
    raw_results = {}  # For raw data before averaging

    print("Starting graph algorithm comparison...")
    print(f"Running {num_runs} trials for each algorithm")
    print(f"Tracking epochs: {track_epochs}")
    print(f"Visualizations will be saved to: {viz_dir}/")
    print(f"  - Each algorithm will have its own subdirectory")
    print(f"  - Each subdirectory will contain {num_runs} PNG files (one per run at epoch 30)")
    print("-" * 80)

    for algo_name, graph_fn in algorithms.items():
        print(f"\nTesting algorithm: {algo_name}")
        algo_results = {epoch: {'first_term': [], 'second_term': [], 'graph_loss': [], 'total_loss': []}
                        for epoch in track_epochs}

        # Initialize raw results storage for this algorithm
        raw_results[algo_name] = {epoch: {'runs': []} for epoch in track_epochs}

        # Create subdirectory for this algorithm's visualizations
        safe_algo_name = algo_name.replace('/', '_').replace('(', '_').replace(')', '_').replace(' ', '_')
        algo_viz_dir = os.path.join(viz_dir, safe_algo_name)
        os.makedirs(algo_viz_dir, exist_ok=True)

        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}")

            # Generate fresh dataset for each run
            dataloader = generate_sample_data(num_samples=1024, batch_size=64)

            # Create model with specific graph function
            model = GraphAutoEncoderComparison(
                input_dim=5,
                output_dim=3,
                hidden_dim=128,
                graph_fn=graph_fn
            ).to(device)

            # Train and track metrics
            run_metrics, graph_data = train_and_track_metrics(
                model, dataloader, epochs, lr, track_epochs
            )

            # Visualize the graph data from epoch 30 for this run
            if graph_data is not None:
                viz_path = os.path.join(algo_viz_dir, f'run_{run+1:03d}_epoch30.png')
                visualize_latent_graph(
                    graph_data['latent'][:, :2],  # Use only first 2 dimensions
                    graph_data['edge_index'],
                    graph_data['edge_attr'],
                    f'{algo_name} - Run {run+1}',
                    viz_path
                )

            # Store raw results for each run
            for epoch in track_epochs:
                raw_results[algo_name][epoch]['runs'].append({
                    'run': run + 1,
                    'first_term': run_metrics[epoch]['first_term'],
                    'second_term': run_metrics[epoch]['second_term'],
                    'graph_loss': run_metrics[epoch]['graph_loss'],
                    'total_loss': run_metrics[epoch]['total_loss']
                })

                # Accumulate for averaging
                for metric_name in ['first_term', 'second_term', 'graph_loss', 'total_loss']:
                    algo_results[epoch][metric_name].append(run_metrics[epoch][metric_name])

        print(f"  Saved {num_runs} visualizations to {algo_viz_dir}/")

        # Calculate averages and store
        for epoch in track_epochs:
            avg_result = {
                'algorithm': algo_name,
                'epoch': epoch,
                'first_term_mean': np.mean(algo_results[epoch]['first_term']),
                'second_term_mean': np.mean(algo_results[epoch]['second_term']),
                'graph_loss_mean': np.mean(algo_results[epoch]['graph_loss']),
                'total_loss_mean': np.mean(algo_results[epoch]['total_loss']),
                'first_term_std': np.std(algo_results[epoch]['first_term']),
                'second_term_std': np.std(algo_results[epoch]['second_term']),
                'graph_loss_std': np.std(algo_results[epoch]['graph_loss']),
                'total_loss_std': np.std(algo_results[epoch]['total_loss'])
            }
            all_results.append(avg_result)

    # Save results
    # Save averaged results
    averaged_results_dict = {
        'parameters': {
            'num_runs': num_runs,
            'epochs': epochs,
            'learning_rate': lr,
            'tracked_epochs': track_epochs,
            'timestamp': timestamp,
            'visualization_directory': viz_dir
        },
        'results': all_results
    }

    averaged_filename = f'graph_comparison_averaged_{timestamp}.json'
    with open(averaged_filename, 'w') as f:
        json.dump(averaged_results_dict, f, indent=2)

    # Save raw results
    raw_results_dict = {
        'parameters': {
            'num_runs': num_runs,
            'epochs': epochs,
            'learning_rate': lr,
            'tracked_epochs': track_epochs,
            'timestamp': timestamp,
            'visualization_directory': viz_dir
        },
        'raw_data': raw_results
    }

    raw_filename = f'graph_comparison_raw_{timestamp}.json'
    with open(raw_filename, 'w') as f:
        json.dump(raw_results_dict, f, indent=2)

    # Print summary table
    print("\n" + "=" * 120)
    print("GRAPH ALGORITHM COMPARISON RESULTS (AVERAGED)")
    print("=" * 120)

    # Create a simple display of results
    for result in all_results:
        if result['epoch'] in [1, 10, 20, 30]:  # Show subset of epochs for clarity
            print(f"{result['algorithm']:<25} Epoch {result['epoch']:>2}: "
                  f"Total Loss = {result['total_loss_mean']:.6f} (±{result['total_loss_std']:.6f})")

    print(f"\nResults saved to:")
    print(f"  - {averaged_filename} (averaged results by epoch)")
    print(f"  - {raw_filename} (raw data before averaging)")
    print(f"  - {viz_dir}/ (latent space graph visualizations)")
    print(f"    └── Each algorithm has its own subdirectory with 100 visualizations (one per run)")
    print(f"\nTotal visualizations created: {len(algorithms) * num_runs} PNG files")


if __name__ == "__main__":
    run_comparison()