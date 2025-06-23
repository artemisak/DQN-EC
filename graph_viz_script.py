import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from pettingzoo.mpe import simple_speaker_listener_v4

import matplotlib.pyplot as plt
import networkx as nx
import os


# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ImprovedGraphAutoEncoder(nn.Module):
    def __init__(self, input_dim=12, output_dim=3, hidden_dim=64):
        super(ImprovedGraphAutoEncoder, self).__init__()

        # Encoder MLP
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # GCN layers
        self.gcn1 = GATv2Conv(in_channels=-1, out_channels=hidden_dim, edge_dim=1)
        self.gcn2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=1)
        self.gcn3 = GATv2Conv(in_channels=hidden_dim, out_channels=output_dim, edge_dim=1)

        # Skip connection layer
        self.skip_connection = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Head
        self.head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.alpha = 0.1

    def create_gabriel_graph(self, points):
        """Create Gabriel graph with minimal preprocessing to preserve information"""
        num_points = points.shape[0]
        edge_indices = []
        edge_attrs = []

        # Convert to numpy for easier Gabriel graph checks
        points_np = points.detach().cpu().numpy()

        for i in range(num_points):
            for j in range(i + 1, num_points):
                midpoint = (points_np[i] + points_np[j]) / 2
                radius_sq = np.sum((points_np[i] - midpoint) ** 2)

                is_gabriel = True
                for k in range(num_points):
                    if k != i and k != j:
                        dist_sq = np.sum((points_np[k] - midpoint) ** 2)
                        if dist_sq < radius_sq:
                            is_gabriel = False
                            break

                if is_gabriel:
                    dist = torch.norm(points[i] - points[j])
                    edge_indices.append([i, j])
                    edge_indices.append([j, i])
                    edge_attrs.append(dist)
                    edge_attrs.append(dist)

        # Ensure there's at least one edge
        if not edge_indices:
            distances = torch.cdist(points, points)
            mask = torch.ones_like(distances, dtype=torch.bool)
            mask.fill_diagonal_(False)
            min_dist, min_indices = torch.min(distances + ~mask * 1e10, dim=1)
            min_dist_idx = torch.argmin(min_dist)
            min_pair_idx = min_indices[min_dist_idx]
            dist = torch.norm(points[min_dist_idx] - points[min_pair_idx])

            edge_indices.append([int(min_dist_idx), int(min_pair_idx)])
            edge_indices.append([int(min_pair_idx), int(min_dist_idx)])
            edge_attrs.append(dist)
            edge_attrs.append(dist)

        # Convert to tensors
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long, device=points.device).t()
            edge_attr = torch.stack(edge_attrs)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=points.device)
            edge_attr = torch.zeros((0, 1), dtype=torch.float, device=points.device)

        return edge_index, edge_attr

    def forward(self, batch):
        batch_size = batch.size(0)

        # Process each sample in the batch individually
        reconstructed_list = []
        latent_list = []
        edge_index_list = []
        edge_attr_list = []

        for idx in range(batch_size):
            # Encode each vector to latent vector
            sample_input = batch[idx]
            latent = self.encoder(sample_input)

            # Create a betta-skeleton graph (special case - Gabriel Graph)
            edge_index, edge_attr = self.create_gabriel_graph(latent)

            # Create PyTorch Geometric Data object
            data = Data(x=latent[:, 2].reshape(-1, 1), edge_index=edge_index, edge_attr=edge_attr)

            # Decode back to original space with GCN
            x1 = F.relu(self.gcn1(data.x, data.edge_index, data.edge_attr))
            x2 = F.relu(self.gcn2(x1, data.edge_index, data.edge_attr))
            gcn_output = self.gcn3(x2, data.edge_index, data.edge_attr)

            # Add skip connection from encoder output
            combined_features = gcn_output + self.alpha * self.skip_connection(latent)

            # Head layer
            reconstructed = self.head(combined_features)

            # Store results
            reconstructed_list.append(reconstructed)
            latent_list.append(latent)
            edge_index_list.append(edge_index)
            edge_attr_list.append(edge_attr)

        # Stack results
        reconstructed_batch = torch.stack(reconstructed_list)
        latent_batch = torch.stack(latent_list)

        return batch, reconstructed_batch, latent_batch, edge_index_list, edge_attr_list


def draw_graph(latent_points, edge_index, edge_attr, title="Gabriel Graph", save_dir="graphs"):
    """
    Draw a graph visualization of the latent space points and their connections
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert tensors to numpy for plotting
    if hasattr(latent_points, 'detach'):
        points = latent_points.detach().cpu().numpy()
    else:
        points = latent_points
        
    if hasattr(edge_index, 'detach'):
        edges = edge_index.detach().cpu().numpy()
    else:
        edges = edge_index
        
    if hasattr(edge_attr, 'detach'):
        edge_weights = edge_attr.detach().cpu().numpy()
    else:
        edge_weights = edge_attr
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes with positions
    num_points = points.shape[0]
    for i in range(num_points):
        G.add_node(i, pos=(points[i, 0], points[i, 1]))
    
    # Add edges if they exist
    if edges.shape[1] > 0:
        for i in range(0, edges.shape[1], 2):  # Skip duplicate edges (undirected)
            node1, node2 = edges[0, i], edges[1, i]
            weight = edge_weights[i] if len(edge_weights) > i else 1.0
            G.add_edge(node1, node2, weight=weight)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw the graph
    if G.number_of_edges() > 0:
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.6, width=2, edge_color='gray')
        
        # Draw edge labels (distances)
        if len(edge_weights) > 0:
            edge_labels = {}
            for i, (u, v) in enumerate(G.edges()):
                if i < len(edge_weights):
                    edge_labels[(u, v)] = f'{edge_weights[i]:.2f}'
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=800, alpha=0.8)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Add z-coordinate as text annotations
    for i, (x, y) in pos.items():
        plt.annotate(f'z={points[i, 2]:.2f}', 
                    (x, y), xytext=(5, 5), 
                    textcoords='offset points', 
                    fontsize=10, alpha=0.8, 
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
    
    plt.title(f'{title}\nNodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}', fontsize=14)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Latent Dimension 1', fontsize=12)
    plt.ylabel('Latent Dimension 2', fontsize=12)
    
    # Save the plot
    filename = f'{title.lower().replace(" ", "_").replace(":", "_").replace(",", "_")}.png'
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), 
                dpi=200, bbox_inches='tight')
    plt.show()  # Display the plot
    plt.close()  # Close the figure to free memory
    
    return G


def load_model(model_path, device):
    """Load the trained model from checkpoint"""
    # Create model instance
    model = ImprovedGraphAutoEncoder().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {model_path}")
    print(f"Final training loss: {checkpoint['final_loss']:.6f}")
    print(f"Epochs trained: {checkpoint['epochs_trained']}")
    
    return model


def sample_environment_data(num_samples=5):
    """Sample a few messages from the environment"""
    env = simple_speaker_listener_v4.parallel_env(max_cycles=100)
    env.reset()
    observations = []
    
    def create_observation_pair(message):
        """Create both observation variants for a message"""
        # First observation (agent_type=0)
        obs1 = torch.zeros(12, 12)
        obs1[:, :11] = torch.eye(12, 11)
        obs1[11, 11] = 0  # agent_type
        obs1[0:8, 11] = torch.tensor(message[:8])  # velocity and landmarks
        obs1[8:11, 11] = -1  # masked out
        
        # Second observation (agent_type=1) 
        obs2 = torch.zeros(12, 12)
        obs2[:, :11] = torch.eye(12, 11)
        obs2[11, 11] = 1  # agent_type
        obs2[0:8, 11] = -1  # masked out
        obs2[8:11, 11] = torch.tensor(message[8:])  # target flags
        
        return [obs1, obs2]
    
    sample_count = 0
    while env.agents and sample_count < num_samples:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, *_ = env.step(actions)
        
        if 'listener_0' in obs:
            pair = create_observation_pair(obs['listener_0'])
            observations.extend(pair)
            sample_count += 1
    
    env.close()
    
    return torch.stack(observations).to(device)


def visualize_model_predictions(model, samples, save_dir="inference_graphs"):
    """Run inference on samples and visualize the resulting graphs"""
    model.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        # Process all samples at once
        original, reconstructed, latent_batch, edge_index_list, edge_attr_list = model(samples)
        
        print("="*60)
        print("MODEL INFERENCE RESULTS")
        print("="*60)
        
        # Visualize each sample
        for i in range(samples.shape[0]):
            print(f"\nSample {i+1}:")
            print("-" * 40)
            
            print("Original observation shape:", original[i].shape)
            print("Reconstructed observation shape:", reconstructed[i].shape)
            print("Latent representation shape:", latent_batch[i].shape)
            print("Number of edges in graph:", edge_index_list[i].shape[1])
            
            # Calculate reconstruction error
            recon_error = F.mse_loss(original[i], reconstructed[i]).item()
            print(f"Reconstruction error (MSE): {recon_error:.6f}")
            
            # Print latent coordinates
            print("Latent coordinates:")
            for j, point in enumerate(latent_batch[i]):
                print(f"  Node {j}: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")
            
            # Visualize the graph
            agent_type = "Listenet" if original[i][11, 11] == 0 else "Speaker"
            title = f"Sample {i+1} - {agent_type} Agent"
            
            draw_graph(
                latent_batch[i], 
                edge_index_list[i], 
                edge_attr_list[i],
                title=title,
                save_dir=save_dir
            )
            
            print(f"Graph visualization saved for {title}")


def main():
    # Model filename
    model_path = "graph_autoencoder_20250623_210243.pth"
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please make sure the model file is in the current directory.")
        return
    
    print("Loading trained model...")
    model = load_model(model_path, device)
    
    print("\nSampling data from environment...")
    samples = sample_environment_data(num_samples=5)
    print(f"Sampled {samples.shape[0]} observations")
    
    print("\nRunning inference and generating visualizations...")
    visualize_model_predictions(model, samples)
    
    print("\nVisualization complete! Check the 'inference_graphs' directory for saved plots.")


if __name__ == "__main__":
    main()