import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import distance
import math
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class GraphAutoEncoder(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, latent_dim=2, gcn_hidden_dim=32):
        super(GraphAutoEncoder, self).__init__()
        
        # Encoder: (3) -> (2)
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # GCN layers using PyTorch Geometric
        self.gcn1 = GCNConv(latent_dim, gcn_hidden_dim)
        self.gcn2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
        
        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(gcn_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
    
    def preprocess(self, x):
        """
        Transform (batch_size, 8) to (batch_size, 8, 3) where each vector is [0, value, index]
        """
        batch_size = x.size(0)
        preprocessed = torch.zeros(batch_size, 8, 3, device=x.device)
        
        for i in range(8):
            preprocessed[:, i, 0] = 0.0  # First element always 0
            preprocessed[:, i, 1] = x[:, i]  # Original value
            preprocessed[:, i, 2] = float(i)  # Position/index
        
        return preprocessed
    
    def create_gabriel_graph(self, points):
        """
        Create a Gabriel graph from 2D points
        Gabriel graph: points i and j have an edge if no other point is inside 
        the circle with diameter (i,j)
        
        Args:
            points: tensor of shape (num_points, 2)
            
        Returns:
            edge_index: tensor of shape (2, num_edges) for PyTorch Geometric
            edge_attr: tensor of shape (num_edges, 1) with edge weights
            edge_weights: list of all edge weights (for loss computation)
        """
        num_points = points.shape[0]
        edge_indices = []
        edge_attrs = []
        edge_weights = []
        
        # Convert to numpy for easier computation
        points_np = points.detach().cpu().numpy()
        
        for i in range(num_points):
            for j in range(i+1, num_points):
                # Calculate midpoint
                midpoint = (points_np[i] + points_np[j]) / 2
                
                # Calculate squared radius
                radius_sq = np.sum((points_np[i] - midpoint) ** 2)
                
                # Check if any other point is inside the circle
                is_gabriel = True
                for k in range(num_points):
                    if k != i and k != j:
                        dist_sq = np.sum((points_np[k] - midpoint) ** 2)
                        if dist_sq < radius_sq:
                            is_gabriel = False
                            break
                
                if is_gabriel:
                    # Euclidean distance for edge weight
                    weight = float(np.sqrt(np.sum((points_np[i] - points_np[j]) ** 2)))
                    
                    # Add edge in both directions (undirected graph)
                    edge_indices.append([i, j])
                    edge_indices.append([j, i])
                    
                    edge_attrs.append([weight])
                    edge_attrs.append([weight])
                    
                    edge_weights.append(torch.tensor(weight, device=points.device))
        
        # Ensure there's at least one edge (connect closest points if no edges)
        if not edge_weights:
            distances = torch.cdist(points, points)
            mask = torch.ones_like(distances, dtype=torch.bool)
            mask.fill_diagonal_(False)
            min_dist, min_indices = torch.min(distances + ~mask * 1e10, dim=1)
            min_dist_idx = torch.argmin(min_dist)
            min_pair_idx = min_indices[min_dist_idx]
            
            min_dist_val = min_dist[min_dist_idx].item()
            
            # Add edge in both directions
            edge_indices.append([int(min_dist_idx), int(min_pair_idx)])
            edge_indices.append([int(min_pair_idx), int(min_dist_idx)])
            
            edge_attrs.append([min_dist_val])
            edge_attrs.append([min_dist_val])
            
            edge_weights.append(torch.tensor(min_dist_val, device=points.device))
        
        # Convert to PyTorch tensors
        if edge_indices:
            edge_index = torch.tensor(edge_indices, device=points.device).t()  # Shape: [2, num_edges]
            edge_attr = torch.tensor(edge_attrs, device=points.device)  # Shape: [num_edges, 1]
        else:
            # Empty graph (shouldn't happen due to our fail-safe)
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=points.device)
            edge_attr = torch.zeros((0, 1), device=points.device)
        
        return edge_index, edge_attr, edge_weights
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Step 1: Preprocess input from (batch_size, 8) to (batch_size, 8, 3)
        x_preprocessed = self.preprocess(x)
        
        # Process each sample in the batch individually
        reconstructed_list = []
        latent_list = []
        edge_index_list = []
        edge_weights_list = []
        
        for b in range(batch_size):
            # Step 2: Encode each (3) vector to (2) vector
            sample_input = x_preprocessed[b]  # Shape: (8, 3)
            latent = self.encoder(sample_input)  # Shape: (8, 2)
            
            # Step 3: Construct Gabriel graph
            edge_index, edge_attr, edge_weights = self.create_gabriel_graph(latent)
            
            # Create PyTorch Geometric Data object
            data = Data(x=latent, edge_index=edge_index, edge_attr=edge_attr)
            
            # Step 4: Process with GCN
            x1 = F.relu(self.gcn1(data.x, data.edge_index, data.edge_attr.squeeze(-1) if data.edge_attr.size(0) > 0 else None))
            x2 = self.gcn2(x1, data.edge_index, data.edge_attr.squeeze(-1) if data.edge_attr.size(0) > 0 else None)
            
            # Step 5: Pool (mean of node features)
            # Using global_mean_pool would be better with batched data
            pooled = torch.mean(x2, dim=0, keepdim=True)  # Shape: (1, gcn_hidden_dim)
            
            # Step 6: Decode back to original space
            node_features = pooled.repeat(8, 1)  # Shape: (8, gcn_hidden_dim)
            decoded = self.decoder(node_features)  # Shape: (8, 3)
            
            # Extract values from middle column to reconstruct original vector
            reconstructed = decoded[:, 1]  # Shape: (8)
            
            # Store results
            reconstructed_list.append(reconstructed)
            latent_list.append(latent)
            edge_index_list.append(edge_index)
            edge_weights_list.append(edge_weights)
        
        # Stack results
        reconstructed_batch = torch.stack(reconstructed_list)  # Shape: (batch_size, 8)
        latent_batch = torch.stack(latent_list)  # Shape: (batch_size, 8, 2)
        
        return reconstructed_batch, latent_batch, edge_index_list, edge_weights_list

# We're now using torch_geometric.nn.GCNConv instead of our custom GCNLayer

# Loss functions
def cosine_distance_loss(original, reconstructed):
    """Cosine distance between original and reconstructed vectors"""
    batch_size = original.shape[0]
    total_loss = 0.0
    
    for i in range(batch_size):
        cos_sim = F.cosine_similarity(original[i].unsqueeze(0), reconstructed[i].unsqueeze(0), dim=1)
        total_loss += (1.0 - cos_sim).item()
    
    return torch.tensor(total_loss / batch_size, device=original.device, requires_grad=True)

def graph_l1_loss(edge_weights_list):
    """L1 measure for graph edges (sum of Euclidean distances)"""
    if not edge_weights_list:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    total_loss = 0.0
    count = 0
    
    for edges in edge_weights_list:
        if edges:
            total_loss = sum(edges)
            count += len(edges)
    
    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return total_loss / count

# Generate sample data
def generate_sample_data(num_samples=1000, batch_size=32):
    """Generate random input data"""
    data = torch.randn(num_samples, 8) * 5.0  # Random values
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, data

# Visualization function
def visualize_latent_space(model, samples, num_samples=5):
    """Visualize 2D latent space with Gabriel graph edges"""
    model.eval()
    with torch.no_grad():
        samples = samples[:num_samples].to(device)
        _, latent, edge_index_list, _ = model(samples)
    
    fig, axs = plt.subplots(1, num_samples, figsize=(15, 3))
    if num_samples == 1:
        axs = [axs]
    
    for i in range(num_samples):
        # Get coordinates and edge indices
        coords = latent[i].cpu().numpy()
        edge_index = edge_index_list[i].cpu().numpy()
        
        # Plot nodes
        axs[i].scatter(coords[:, 0], coords[:, 1], c='blue', s=100, zorder=5)
        
        # Plot edges
        if edge_index.size > 0:
            # Get unique edges (avoid duplicates from undirected graph)
            edges_set = set()
            for j in range(edge_index.shape[1]):
                src, dst = edge_index[0, j], edge_index[1, j]
                if src < dst:  # Only consider one direction for visualization
                    edges_set.add((src, dst))
            
            # Plot each edge
            for src, dst in edges_set:
                axs[i].plot([coords[src, 0], coords[dst, 0]], 
                         [coords[src, 1], coords[dst, 1]], 'k-', alpha=0.5, zorder=1)
        
        # Add node labels
        for j, (x, y) in enumerate(coords):
            axs[i].text(x, y, str(j), fontsize=10, ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        axs[i].set_title(f'Sample {i+1}')
        axs[i].set_aspect('equal')
        axs[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('latent_space_visualization.png')
    plt.show()

# Training function
def train_model(model, dataloader, test_samples, epochs=100, lr=0.001, graph_weight=0.1):
    """Train the autoencoder model"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    train_losses = []
    rec_losses = []
    graph_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_rec_loss = 0.0
        epoch_graph_loss = 0.0
        
        for batch_idx, (batch,) in enumerate(dataloader):
            batch = batch.to(device)
            
            # Forward pass
            reconstructed, latent, adj_matrices, edge_weights = model(batch)
            
            # Calculate losses
            rec_loss = cosine_distance_loss(batch, reconstructed)
            g_loss = graph_l1_loss(edge_weights)
            total_loss = rec_loss + graph_weight * g_loss
            
            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Record losses
            epoch_loss += total_loss.item()
            epoch_rec_loss += rec_loss.item()
            epoch_graph_loss += g_loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch: {batch_idx+1}, '
                      f'Loss: {total_loss.item():.4f}, Rec Loss: {rec_loss.item():.4f}, '
                      f'Graph Loss: {g_loss.item():.4f}')
        
        # Calculate average losses
        avg_loss = epoch_loss / len(dataloader)
        avg_rec_loss = epoch_rec_loss / len(dataloader)
        avg_graph_loss = epoch_graph_loss / len(dataloader)
        
        train_losses.append(avg_loss)
        rec_losses.append(avg_rec_loss)
        graph_losses.append(avg_graph_loss)
        
        # Print epoch summary
        print(f'Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}, '
              f'Avg Rec Loss: {avg_rec_loss:.4f}, Avg Graph Loss: {avg_graph_loss:.4f}')
        
        # Update learning rate
        scheduler.step()
        
        # Visualize latent space periodically
        if (epoch + 1) % 20 == 0 or epoch == 0:
            visualize_latent_space(model, test_samples, num_samples=3)
    
    # Plot training losses
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(rec_losses)
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(graph_losses)
    plt.title('Graph Loss')
    plt.xlabel('Epoch')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_losses.png')
    plt.show()
    
    return model

# Main function
def main():
    # Make sure we have PyTorch Geometric
    try:
        import torch_geometric
        print(f"Using PyTorch Geometric version: {torch_geometric.__version__}")
    except ImportError:
        print("PyTorch Geometric not found. Please install with:")
        print("pip install torch-geometric")
        return
    
    # Generate dataset
    dataloader, samples = generate_sample_data(num_samples=1000, batch_size=32)
    
    # Create model
    model = GraphAutoEncoder().to(device)
    print(model)
    
    # Train model
    trained_model = train_model(
        model, 
        dataloader, 
        samples, 
        epochs=100, 
        lr=0.001, 
        graph_weight=0.1
    )
    
    # Visualize final latent space
    visualize_latent_space(trained_model, samples, num_samples=5)
    
    # Save the model
    torch.save({
        'model_state_dict': trained_model.state_dict(),
    }, 'graph_autoencoder_model.pth')
    
    print("Training complete. Model saved to graph_autoencoder_model.pth")

if __name__ == "__main__":
    main()
