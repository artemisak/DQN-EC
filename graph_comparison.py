import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from datetime import datetime
import json

from graphs_wrapper import (
    create_delaunay_graph,
    create_beta_skeleton_graph,
    create_gabriel_graph,
    create_dal_graph,
    create_knn_graph,
    create_full_connected_graph
)
from graph_autoencoder import generate_sample_data

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


def train_and_track_metrics(model, dataloader, epochs, lr, track_epochs):
    """Train model and track metrics at specific epochs"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    metrics = {epoch: {} for epoch in track_epochs}

    for epoch in range(epochs):
        model.train()
        epoch_metrics = {
            'first_term': [],
            'second_term': [],
            'graph_loss': [],
            'total_loss': []
        }

        for _, (batch,) in enumerate(dataloader):
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

    return metrics


def run_comparison():
    """Run comparison of different graph algorithms"""

    # Define algorithms to test
    algorithms = {
        'DAL': create_dal_graph,
        'Beta Skeleton (β=1.7)': lambda points: create_beta_skeleton_graph(points, beta=1.7),
        'Beta Skeleton (β=1.0)': create_gabriel_graph,
        'Fully Connected': create_full_connected_graph,
        'Delaunay': create_delaunay_graph,
        'kNN': create_knn_graph,
    }

    # Parameters
    num_runs = 3
    epochs = 30
    lr = 0.0025
    track_epochs = [1, 5, 10, 15, 20, 25, 30]

    # Results storage
    all_results = []  # For averaged results
    raw_results = {}  # For raw data before averaging

    print("Starting graph algorithm comparison...")
    print(f"Running {num_runs} trials for each algorithm")
    print(f"Tracking epochs: {track_epochs}")
    print("-" * 80)

    for algo_name, graph_fn in algorithms.items():
        print(f"\nTesting algorithm: {algo_name}")
        algo_results = {epoch: {'first_term': [], 'second_term': [], 'graph_loss': [], 'total_loss': []}
                        for epoch in track_epochs}

        # Initialize raw results storage for this algorithm
        raw_results[algo_name] = {epoch: {'runs': []} for epoch in track_epochs}

        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}")

            # Generate fresh dataset for each run
            dataloader = generate_sample_data(num_samples=1024, batch_size=64)

            # Create model with specific graph function
            model = GraphAutoEncoderComparison(
                input_dim=5,
                output_dim=3,
                hidden_dim=64,
                graph_fn=graph_fn
            ).to(device)

            # Train and track metrics
            run_metrics = train_and_track_metrics(
                model, dataloader, epochs, lr, track_epochs
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save averaged results
    averaged_results_dict = {
        'parameters': {
            'num_runs': num_runs,
            'epochs': epochs,
            'learning_rate': lr,
            'tracked_epochs': track_epochs,
            'timestamp': timestamp
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
            'timestamp': timestamp
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


if __name__ == "__main__":
    run_comparison()