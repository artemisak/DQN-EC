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
from datetime import datetime


# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

class GraphAutoEncoder(nn.Module):
    def __init__(self, input_dim=5, output_dim=3, hidden_dim=64):
        super(GraphAutoEncoder, self).__init__()

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
        self.gcn1 = GATv2Conv(in_channels=1, out_channels=hidden_dim)
        self.gcn2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim)

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

        reconstructed_labels = []
        reconstructed_values = []
        latent_list = []
        edge_index_list = []
        edge_attr_list = []

        for _, obs in enumerate(batch):
            # Encode each vector to latent vector
            latent = self.encoder(obs)

            # Create a betta-skeleton graph (with beta = 1 we got the special case of Gabriel Graph)
            edge_index, edge_attr = self.create_gabriel_graph(latent)

            # Create PyTorch Geometric Data object
            graph = Data(x=latent[:, 2].reshape(-1, 1), edge_index=edge_index)

            x1 = F.relu(self.gcn1(graph.x, graph.edge_index))
            x2 = F.relu(self.gcn2(x1, graph.edge_index))

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


# Improved reconstruction loss with dimension-specific weighting
def reconstruction_loss(true_distribution, predicted_logits, true_values, predicted_values, epoch, total_epochs):
    """
    Reconstruction loss
    """
    labels_kl = F.kl_div(F.log_softmax(predicted_logits, dim=-1), F.softmax(true_distribution, dim=-1), reduction='batchmean')
    values_mse = F.l1_loss(predicted_values, true_values)

    progress = epoch / total_epochs
    weight1 = 1.0 - 0.9 * progress
    weight2 = 0.1 + 0.9 * progress

    return  weight1 * labels_kl + weight2 * values_mse

def l1_loss(edge_attr_list):
    """
    L1 loss for the edges on the graph
    """
    return torch.norm(torch.cat(edge_attr_list), p=1)


def draw_graph(latent_points, edge_index, edge_attr, title="Gabriel Graph", save_dir="graphs"):
    """
    Draw a graph visualization of the latent space points and their connections
    
    Args:
        latent_points: Tensor of shape (num_points, 3) containing 3D coordinates
        edge_index: Tensor of shape (2, num_edges) containing edge connections
        edge_attr: Tensor of edge attributes (distances)
        title: Title for the plot
        save_dir: Directory to save the plots
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
    plt.figure(figsize=(10, 8))
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw the graph
    if G.number_of_edges() > 0:
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.6, width=1.5, edge_color='gray')
        
        # Draw edge labels (distances)
        if len(edge_weights) > 0:
            edge_labels = {}
            for i, (u, v) in enumerate(G.edges()):
                if i < len(edge_weights):
                    edge_labels[(u, v)] = f'{edge_weights[i]:.2f}'
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, alpha=0.8)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add z-coordinate as text annotations
    for i, (x, y) in pos.items():
        plt.annotate(f'z={points[i, 2]:.2f}', 
                    (x, y), xytext=(5, 5), 
                    textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    plt.title(f'{title}\nNodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    
    # Save the plot
    filename = f'{title.lower().replace(" ", "_").replace(":", "_")}.png'
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), 
                dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    return G


def train_model(model, dataloader, epochs, lr, save_path="trained_model.pth"):
    """Train with only reconstruction loss for better convergence"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for _, (batch,) in enumerate(dataloader):

            # Forward pass
            true_distribution, true_values, predicted_logits, predicted_values, latent_batch, edge_index_list, edge_attr_list = model(batch)

            # Calculate the combined loss function
            loss = reconstruction_loss(true_distribution, predicted_logits, true_values, predicted_values, epoch, epochs)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch % 5 == 0) or (epoch == epochs - 1):
            print(f'Epoch: {epoch}')
            print('='*50, 'Original labels', '='*50)
            print(F.softmax(true_distribution[0], dim=-1))
            print('=' * 50, 'Reconstructed labes', '=' * 50)
            print(F.softmax(predicted_logits[0], dim=-1))
            print('='*50, 'Original values', '='*50)
            print(true_values[0])
            print('=' * 50, 'Reconstructed values', '=' * 50)
            print(predicted_values[0])
            draw_graph(latent_batch[11], edge_index=edge_index_list[11], edge_attr=edge_attr_list[11],
                      title=f"Training Epoch {epoch}", save_dir="training_graphs")

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)

        print(f'Epoch [{epoch + 1}/{epochs}], Avg Loss: {avg_loss:.6f}')

        # Update learning rate based on validation loss
        scheduler.step(avg_loss)

    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_loss': avg_loss,
        'epochs_trained': epochs,
        'model_config': {
            'input_dim': 12,
            'output_dim': 3,
            'hidden_dim': 64
        }
    }, save_path)
    
    print(f"Model saved to {save_path}")


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

# Main function
def main():
    # Create directories for saving outputs
    os.makedirs("training_graphs", exist_ok=True)
    
    # Generate dataset
    dataloader = generate_sample_data()

    # Create model
    model = GraphAutoEncoder().to(device)
    print(model)

    # Create a timestamped model filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"graph_autoencoder_{timestamp}.pth"

    # Hyperparameters for training
    train_model(
        model,
        dataloader,
        epochs=30,
        lr=0.0025,
        save_path=model_filename
    )
    
    print(f"Training completed! Model saved as {model_filename}")

if __name__ == "__main__":
    main()