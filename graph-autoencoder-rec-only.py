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
    def __init__(self, input_dim=3, hidden_dim=64):
        super(ImprovedGraphAutoEncoder, self).__init__()

        # Encoder MLP
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # GCN layers
        self.gcn1 = GATv2Conv(in_channels=-1, out_channels=hidden_dim, edge_dim=1)
        self.gcn2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=1)
        self.gcn3 = GATv2Conv(in_channels=hidden_dim, out_channels=input_dim, edge_dim=1)

        # Skip connection layer
        self.skip_connection = nn.Linear(input_dim, input_dim)

        # Learnable skip connection layer's threshold
        self.skip_connection_alpha = nn.Sigmoid()

        self.alpha = 0.1

    def preprocess(self, x):
        """
        Transform (32, 8) to (32, 8, 3) where each vector is [agent_type, index, value]
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

        # Preprocess input from (batch_size, 8) to (batch_size, 8, 3)
        x_preprocessed = self.preprocess(x)

        # Process each sample in the batch individually
        reconstructed_list = []
        latent_list = []
        edge_index_list = []

        for b in range(batch_size):
            # Encode each (3) vector to latent vector
            sample_input = x_preprocessed[b]  # Shape: (8, 3)
            latent = self.encoder(sample_input)  # Shape: (8, 3)

            # Create a betta-skeleton graph (special case - Gabriel Graph)
            edge_index, edge_attr = self.create_gabriel_graph(latent)

            # Create PyTorch Geometric Data object
            data = Data(x=latent[:, 2].reshape(-1, 1), edge_index=edge_index, edge_attr=edge_attr)

            # Decode back to original space with GCN
            x1 = F.relu(self.gcn1(data.x, data.edge_index, data.edge_attr))
            x2 = F.relu(self.gcn2(x1, data.edge_index, data.edge_attr))
            gcn_output = self.gcn3(x2, data.edge_index, data.edge_attr)

            # Add skip connection from encoder output
            combined_features = gcn_output + self.alpha * self.skip_connection(latent) # Shape: (8, 3)

            # Store results
            reconstructed_list.append(combined_features)
            latent_list.append(latent)  # Store visualization latent for display
            edge_index_list.append(edge_index)

        # Stack results
        reconstructed_batch = torch.stack(reconstructed_list)  # Shape: (32, 8, 3)
        latent_batch = torch.stack(latent_list)  # Shape: (32, 8, 3)

        return x_preprocessed, reconstructed_batch, latent_batch, edge_index_list


# Improved reconstruction loss with dimension-specific weighting
def reconstruction_loss(original, reconstructed):
    """
    Reconstruction loss with separate handling of different dimensions
    """
    # Ensure shapes match
    assert original.shape == reconstructed.shape, f"Shape mismatch: {original.shape} vs {reconstructed.shape}"

    # Calculate MSE for each dimension separately
    dim0_loss = F.mse_loss(original[:, :, 0], reconstructed[:, :, 0])  # Agent type
    dim1_loss = F.mse_loss(original[:, :, 1], reconstructed[:, :, 1])  # Position/index
    dim2_loss = F.mse_loss(original[:, :, 2], reconstructed[:, :, 2])  # Value

    return dim0_loss + dim1_loss + dim2_loss


def train_model(model, dataloader, epochs, lr):
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

            # Calculate only reconstruction loss
            loss = reconstruction_loss(original, reconstructed)

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

    return model


# Generate sample data
def generate_sample_data(num_samples=1000, batch_size=32):
    """Generate random input data"""
    data = torch.randn(num_samples, 8) * 5.0  # Random values
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# Main function
def main():

    # Generate dataset
    dataloader = generate_sample_data(num_samples=1000, batch_size=32)

    # Create model
    model = ImprovedGraphAutoEncoder().to(device)
    print(model)

    # Hyperparameters for training
    trained_model = train_model(
        model,
        dataloader,
        epochs=10,
        lr=0.0025
    )

if __name__ == "__main__":
    main()