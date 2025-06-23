from collections import defaultdict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.patches as mpatches
from dataclasses import dataclass

@dataclass
class NodeDescriptor:
    name: str       # Например: "self_vel-x"
    group: str      # Тип: "self_vel", "landmark"

group_color_map = {
    "self_vel": "#1f77b4",
    "landmark": "#2ca02c",
    "unknown": "#7f7f7f"
}

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
    

    def beta_skeleton_graph(self, points, beta=1):
        n = points.shape[0]
        edges = []
        
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = points[i], points[j]
                d = np.linalg.norm(p1 - p2)
                
                if d < 1e-10:  # Skip identical points
                    continue
                
                is_edge = True
                
                if beta == 1:
                    # Gabriel graph: check if any point lies inside the circle 
                    # with diameter p1-p2
                    center = (p1 + p2) / 2
                    radius = d / 2
                    
                    for k in range(n):
                        if k != i and k != j:
                            pk = points[k]
                            dist_to_center = np.linalg.norm(pk - center)
                            if dist_to_center < radius:  # Strictly inside
                                is_edge = False
                                break
                else:
                    # General beta skeleton
                    if beta < 1e-5:
                        continue
                    
                    # For beta != 1, use the lune-based definition
                    # Two circles of radius d/(2*beta) centered at points that are
                    # distance d*beta/2 from the midpoint along the perpendicular
                    center = (p1 + p2) / 2
                    direction = (p2 - p1) / d  # unit vector along edge
                    perpendicular = np.array([-direction[1], direction[0]])  # perpendicular unit vector
                    
                    offset = d * np.sqrt(1/(4*beta**2) - 1/4) if beta > 1 else 0
                    
                    if beta > 1:
                        # Two circle centers
                        c1 = center + offset * perpendicular
                        c2 = center - offset * perpendicular
                        radius = d / (2 * beta)
                        
                        for k in range(n):
                            if k != i and k != j:
                                pk = points[k]
                                # Point must be outside both circles
                                if (np.linalg.norm(pk - c1) < radius or 
                                    np.linalg.norm(pk - c2) < radius):
                                    is_edge = False
                                    break
                    else:  # beta < 1
                        # Single circle
                        radius = d / (2 * beta)
                        for k in range(n):
                            if k != i and k != j:
                                pk = points[k]
                                if np.linalg.norm(pk - center) > radius:
                                    is_edge = False
                                    break
                
                if is_edge:
                    edges.append([i, j])
                    edges.append([j, i])  # bidirectional
        
        return edges
    
    def forward(self, x):
        batch_size = x.size(0)

        # Preprocess input from (batch_size, 8) to (batch_size, 8, 3)
        x_preprocessed = self.preprocess(x)

        # Process each sample in the batch individually
        reconstructed_list = []
        latent_list = []
        edge_index_list = []
        attn_weights_list = []

        for b in range(batch_size):
            # Encode each (3) vector to latent vector
            sample_input = x_preprocessed[b]  # Shape: (8, 3)
            latent = self.encoder(sample_input)  # Shape: (8, 3)

            # Normalization
            # Center
            mu = torch.mean(latent, dim=0, keepdim=True)
            graph_latent = latent - mu

            # Scaling
            std = torch.std(graph_latent, dim=0)
            graph_latent = graph_latent / (std.unsqueeze(0) + 1e-8)

            # Create a betta-skeleton graph (special case - Gabriel Graph)
            #edge_index, edge_attr = self.create_gabriel_graph(graph_latent)

            # Create a betta-skeleton graph (special case - Gabriel Graph)
            graph_latent_np = graph_latent.detach().cpu().numpy()
            edges = self.beta_skeleton_graph(graph_latent_np)
            edge_index = torch.tensor(edges, dtype=torch.long, device=x.device).t()
            edge_attr = torch.norm(latent[edge_index[0]] - latent[edge_index[1]], dim=1, keepdim=True)

            # Create PyTorch Geometric Data object
            data = Data(x=latent[:, 2].reshape(-1, 1), edge_index=edge_index, edge_attr=edge_attr)

            # Decode back to original space with GCN
            x1 = F.relu(self.gcn1(data.x, data.edge_index, data.edge_attr))
            x2 = F.relu(self.gcn2(x1, data.edge_index, data.edge_attr))
            gcn_output, (final_edge_index, final_attn_weights) = self.gcn3(x2, data.edge_index, data.edge_attr, return_attention_weights=True)

            # Add skip connection from encoder output
            combined_features = gcn_output + self.alpha * self.skip_connection(latent) # Shape: (8, 3)

            # Store results
            reconstructed_list.append(combined_features)
            latent_list.append(graph_latent)  # Store visualization latent for display
            edge_index_list.append(edge_index)
            attn_weights_list.append(final_attn_weights)

        # Stack results
        reconstructed_batch = torch.stack(reconstructed_list)  # Shape: (32, 8, 3)
        latent_batch = torch.stack(latent_list)  # Shape: (32, 8, 3)


        return x_preprocessed, reconstructed_batch, latent_batch, edge_index_list, attn_weights_list


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

def frobenius_inequality_loss(original, reconstructed, epsilon=1.0):
    """
    Implements Frobenius inequality as a soft constraint:
        ||original - reconstructed||_F <= epsilon

    Returns a loss term that is zero if the constraint is satisfied,
    and positive otherwise.

    Args:
        original (torch.Tensor): shape (B, N, D)
        reconstructed (torch.Tensor): shape (B, N, D)
        epsilon (float): upper threshold for Frobenius norm

    Returns:
        torch.Tensor: scalar loss penalizing violation of the inequality
    """
    assert original.shape == reconstructed.shape, "Shape mismatch"

    # Compute Frobenius norm per sample
    frob_norms = torch.norm(original - reconstructed, p='fro', dim=(1, 2))  # shape: (B,)
    
    # Compute violation: max(0, norm - epsilon)
    violations = torch.clamp(frob_norms - epsilon, min=0.0)

    # Mean over batch
    return violations.mean()

def train_model(model, dataloader, epochs, lr, param_schema):
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
            original, reconstructed, latent, edge_index_list, attn_weights_list = model(batch)

            # Calculate only reconstruction loss
            loss = frobenius_inequality_loss(original, reconstructed)

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
        visualize_and_save_gabriel_graph(
            latent_points=latent[0].detach().cpu(),
            edge_index=edge_index_list[0].detach().cpu(),
            attn_weights=attn_weights_list[0].detach().cpu(),
            epoch=epoch,
            param_schema=param_schema
        )
    return model

# Generate sample data
def generate_sample_data(num_samples=1000, batch_size=32):
    """Generate random input data"""
    data = torch.randn(num_samples, 8) * 5.0  # Random values
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def visualize_and_save_gabriel_graph(latent_points, edge_index, attn_weights, epoch, save_dir="graph_visualizations", param_schema=None):
    """
    Визуализация и сохранение Gabriel-графа на диск.
    
    :param latent_points: (8, 3) — координаты узлов
    :param edge_index: (2, num_edges) — рёбра графа
    :param epoch: номер эпохи
    :param save_dir: директория для сохранения изображений
    """
    os.makedirs(save_dir, exist_ok=True)

    G = nx.Graph()
    positions = {}

    coords = []
    for i in range(latent_points.shape[0]):
        x, y = latent_points[i][0].item(), latent_points[i][1].item()
        G.add_node(i)
        positions[i] = (x, y)
        coords.append([x, y])

    coords = np.array(coords)
    edges = edge_index.t().cpu().numpy()
    for src, tgt in edges:
        G.add_edge(src, tgt)

    mst = nx.minimum_spanning_tree(G)

    node_colors = []
    for i in range(len(latent_points)):
        if param_schema:
            group = param_schema[i].group
        else:
            group = "unknown"
        color = group_color_map.get(group, group_color_map["unknown"])
        node_colors.append(color)

    weights = attn_weights.squeeze().cpu().numpy()
    edge_labels = {}
    for (src, tgt), weight in zip(edges, weights):
        G.add_edge(src, tgt, weight=weight)
        edge_labels[(src, tgt)] = float(weight)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Рисуем ноды
    nx.draw_networkx_nodes(
        G,
        pos=positions,
        node_color=node_colors,
        node_size=60,
        ax=ax,
        linewidths=0.8,
        edgecolors="black"
    )

    # Подписи над точками — выше и правее, мелко, без фона
    for i, (x, y) in positions.items():
        ax.annotate(
            str(i),
            (x, y),
            textcoords="offset points",
            xytext=(0, -10),
            ha='center',
            fontsize=7,
            color='black'
        )

    nx.draw_networkx_edges(G, pos=positions, ax=ax, edge_color="#2fe94e", width=1.2)

    # Draw MST
    nx.draw_networkx_edges(mst, pos=positions, ax=ax, edge_color="#CA2171", width=1.6, style="dashed")

    # for src, tgt in edges:
    #     x1, y1 = positions[src]
    #     x2, y2 = positions[tgt]

    #     # Центр окружности — середина отрезка
    #     center_x = (x1 + x2) / 2
    #     center_y = (y1 + y2) / 2

    #     # Радиус = половина расстояния между точками
    #     radius = 0.5 * ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    #     circle = Circle((center_x, center_y), radius, edgecolor='#E94F31', facecolor='none', linestyle='--', linewidth=1.5)
    #     ax.add_patch(circle)

    # vor = Voronoi(coords)
    # voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=1.5, line_alpha=0.6, point_size=0)

    # Сначала создаём стандартные элементы легенды по группам
    legend_elements = [
        mpatches.Patch(color=color, label=group)
        for group, color in group_color_map.items() if group != "unknown"
    ]

    ax.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.08),  # под графиком
        ncol=len(legend_elements),
        fontsize=8,
        frameon=False
    )

    if param_schema:
        max_name_len = max(len(p.name) for p in param_schema)
    else:
        max_name_len = max(len(f"node_{i}") for i in range(len(latent_points)))

    # Минимальная ширина имени, чтобы не было слишком узко
    name_col_width = max(max_name_len, 10)
    num_col_width = max(len(str(len(latent_points) - 1)), 3)  # ширина для номера узла

    # Формируем заголовки с динамической шириной
    header_nodes = f"Nodes:\n{'No.':<{num_col_width}} | {'Name':<{name_col_width}} |     X     |     Y    \n"
    separator_nodes = "-" * (num_col_width + name_col_width + 24) + "\n"
    node_table_text = header_nodes + separator_nodes
    for i in range(len(latent_points)):
        name = param_schema[i].name if param_schema else f"node_{i}"
        x_val = latent_points[i][0].item()
        y_val = latent_points[i][1].item()
        node_table_text += f"{i:<{num_col_width}} | {name:<{name_col_width}} | {x_val:>8.3f} | {y_val:>8.3f}\n"

    header_edges = f"Edges:\n{'From':<{name_col_width}} | {'To':<{name_col_width}} | Weight\n"
    separator_edges = "-" * (name_col_width * 2 + 11) + "\n"
    edge_table_text = header_edges + separator_edges
    for (src, tgt), weight in edge_labels.items():
        src_name = param_schema[src].name if param_schema else f"node_{src}"
        tgt_name = param_schema[tgt].name if param_schema else f"node_{tgt}"
        edge_table_text += f"{src_name:<{name_col_width}} | {tgt_name:<{name_col_width}} | {weight:>7.3f}\n"

    ax.text(
        1.02, 0.95, node_table_text,
        transform=ax.transAxes,
        fontsize=7,
        family='monospace',
        verticalalignment='top',
        horizontalalignment='left'
    )

    ax.text(
        1.02, 0.6, edge_table_text,
        transform=ax.transAxes,
        fontsize=7,
        family='monospace',
        verticalalignment='top',
        horizontalalignment='left'
    )

    ax.set_title(f"Latent Graph — Epoch {epoch + 1}", fontsize=11)
    ax.axis('equal')
    ax.axis('off')
    plt.tight_layout(rect=[0, 0.05, 0.8, 1])

    filename = os.path.join(save_dir, f"epoch{epoch + 1:02d}.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


# Main function
def main():
    # Generate dataset
    dataloader = generate_sample_data(num_samples=1000, batch_size=32)

    # Create model
    model = ImprovedGraphAutoEncoder().to(device)
    print(model)

    param_schema: List[NodeDescriptor] = [
        NodeDescriptor("vel-x", "self_vel"),
        NodeDescriptor("vel-y", "self_vel"),
        NodeDescriptor("landmark-1-rel-x", "landmark"),
        NodeDescriptor("landmark-1-rel-y", "landmark"),
        NodeDescriptor("landmark-2-rel-x", "landmark"),
        NodeDescriptor("landmark-2-rel-y", "landmark"),
        NodeDescriptor("landmark-3-rel-x", "landmark"),
        NodeDescriptor("landmark-3-rel-y", "landmark"),
    ]

    # Hyperparameters for training
    trained_model = train_model(
        model,
        dataloader,
        epochs=10,
        lr=0.0025,
        param_schema=param_schema
    )

if __name__ == "__main__":
    main()