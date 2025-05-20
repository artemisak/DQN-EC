import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data


# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ImprovedGraphAutoEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, latent_dim=3):
        super(ImprovedGraphAutoEncoder, self).__init__()

        # Encoder MLP
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # GCN layers - Process ALL latent dimensions
        self.gcn1 = GATv2Conv(in_channels=latent_dim, out_channels=hidden_dim, edge_dim=1)
        self.gcn2 = GATv2Conv(in_channels=hidden_dim, out_channels=latent_dim, edge_dim=1)

        # Decoder MLP with increased capacity
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Skip connection layers
        self.skip_connection = nn.Linear(latent_dim, latent_dim)

    def preprocess(self, x):
        """
        Transform (batch_size, 8) to (batch_size, 8, 3) where each vector is [agent_type, index, value]
        """
        batch_size = x.size(0)
        preprocessed = torch.zeros(batch_size, 8, 3, device=x.device)

        for i in range(8):
            preprocessed[:, i, 0] = 0.0  # Agent type
            preprocessed[:, i, 1] = float(i)  # Position/index
            preprocessed[:, i, 2] = x[:, i]  # Agent's observation value

        return preprocessed

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
                    dist = torch.norm(points[i] - points[j]).item()
                    edge_indices.append([i, j])
                    edge_indices.append([j, i])
                    edge_attrs.append([dist])
                    edge_attrs.append([dist])

        # Ensure there's at least one edge
        if not edge_indices:
            distances = torch.cdist(points, points)
            mask = torch.ones_like(distances, dtype=torch.bool)
            mask.fill_diagonal_(False)
            min_dist, min_indices = torch.min(distances + ~mask * 1e10, dim=1)
            min_dist_idx = torch.argmin(min_dist)
            min_pair_idx = min_indices[min_dist_idx]
            dist = torch.norm(points[min_dist_idx] - points[min_pair_idx]).item()

            edge_indices.append([int(min_dist_idx), int(min_pair_idx)])
            edge_indices.append([int(min_pair_idx), int(min_dist_idx)])
            edge_attrs.append([dist])
            edge_attrs.append([dist])

        # Convert to tensors
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long, device=points.device).t()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float, device=points.device)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=points.device)
            edge_attr = torch.zeros((0, 1), dtype=torch.float, device=points.device)

        return edge_index, edge_attr

    def forward(self, x):
        batch_size = x.size(0)

        # Step 1: Preprocess input from (batch_size, 8) to (batch_size, 8, 3)
        x_preprocessed = self.preprocess(x)

        # Process each sample in the batch individually
        reconstructed_list = []
        latent_list = []
        edge_index_list = []

        for b in range(batch_size):
            # Step 2: Encode each (3) vector to latent vector
            sample_input = x_preprocessed[b]  # Shape: (8, 3)
            latent = self.encoder(sample_input)  # Shape: (8, latent_dim)

            # IMPORTANT: Store original latent for skip connection
            original_latent = latent.clone()

            # For visualization only - apply mild normalization that doesn't destroy information
            # Center the latent points (important for proper scaling)
            vis_latent = latent - torch.mean(latent, dim=0, keepdim=True)

            # Gentle scaling for visualization only - doesn't affect reconstruction path
            std = torch.std(vis_latent, dim=0) + 1e-8
            vis_latent = vis_latent / std.unsqueeze(0) * 1.0

            # Use the visualization latent only for graph creation
            edge_index, edge_attr = self.create_gabriel_graph(vis_latent)

            # IMPORTANT: Use ALL latent dimensions in the GCN (not just the 3rd)
            # Create PyTorch Geometric Data object with full latent space
            data = Data(x=latent, edge_index=edge_index, edge_attr=edge_attr)

            # Step 4: Process with GCN
            x1 = F.relu(self.gcn1(data.x, data.edge_index, data.edge_attr))
            gcn_output = self.gcn2(x1, data.edge_index, data.edge_attr)

            # Add skip connection from encoder output
            # This allows information to flow directly to the decoder
            skip_processed = self.skip_connection(original_latent)
            combined_features = gcn_output + skip_processed

            # Step 5: Decode back to original space
            reconstructed = self.decoder(combined_features)  # Shape: (8, 3)

            # Store results
            reconstructed_list.append(reconstructed)
            latent_list.append(vis_latent)  # Store visualization latent for display
            edge_index_list.append(edge_index)

        # Stack results
        reconstructed_batch = torch.stack(reconstructed_list)  # Shape: (batch_size, 8, 3)
        latent_batch = torch.stack(latent_list)  # Shape: (batch_size, 8, latent_dim)

        return x_preprocessed, reconstructed_batch, latent_batch, edge_index_list


# Improved reconstruction loss with dimension-specific weighting
def improved_reconstruction_loss(original, reconstructed):
    """
    Weighted reconstruction loss with separate handling of different dimensions
    """
    # Ensure shapes match
    assert original.shape == reconstructed.shape, f"Shape mismatch: {original.shape} vs {reconstructed.shape}"

    # Calculate MSE for each dimension separately
    # This allows us to weight different dimensions differently
    dim0_loss = F.mse_loss(original[:, :, 0], reconstructed[:, :, 0])  # Agent type
    dim1_loss = F.mse_loss(original[:, :, 1], reconstructed[:, :, 1])  # Position/index
    dim2_loss = F.mse_loss(original[:, :, 2], reconstructed[:, :, 2])  # Value

    # Weight the dimensions differently if needed
    # We care more about reconstructing values than agent types
    weighted_loss = 0.5 * dim0_loss + 1.0 * dim1_loss + 2.0 * dim2_loss

    return weighted_loss


def train_model(model, dataloader, test_samples, epochs=100, lr=0.001):
    """Train with only reconstruction loss for better convergence"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (batch,) in enumerate(dataloader):
            batch = batch.to(device)

            # Forward pass
            original, reconstructed, latent, edge_index_list = model(batch)
            if epoch == epochs - 1:
                print(original, reconstructed)

            # Calculate only reconstruction loss
            loss = improved_reconstruction_loss(original, reconstructed)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)

        print(f'Epoch [{epoch + 1}/{epochs}], Avg Loss: {avg_loss:.6f}')

        # Update learning rate based on validation loss
        scheduler.step(avg_loss)

    return model, train_losses


# Generate sample data
def generate_sample_data(num_samples=1000, batch_size=32):
    """Generate random input data"""
    data = torch.randn(num_samples, 8) * 5.0  # Random values
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, data


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
    model = ImprovedGraphAutoEncoder().to(device)
    print(model)

    # FIX 9: Adjust hyperparameters for training
    trained_model = train_model(
        model,
        dataloader,
        samples,
        epochs=30,
        lr=0.0025
    )

    print("Training complete. Model saved to graph_autoencoder_model.pth")

if __name__ == "__main__":
    main()