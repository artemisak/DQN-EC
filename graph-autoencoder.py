from dataclasses import dataclass
from typing import List

from matplotlib.patches import Circle
from networkx.drawing import draw_networkx_edges
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.patches as mpatches
from dataclasses import dataclass
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

    def build_beta_skeleton(self, points, beta: float):
        num_points = points.shape[0]
        edge_indices = []
        edge_attrs = []

        points_np = points.detach().cpu().numpy()

        for i in range(num_points):
            for j in range(i + 1, num_points):
                pi = points_np[i]
                pj = points_np[j]

                midpoint = (pi + pj) / 2
                dij = np.linalg.norm(pi - pj)

                is_edge = True
                for k in range(num_points):
                    if k == i or k == j:
                        continue
                    pk = points_np[k]

                    if beta == 0:
                        # Relative Neighborhood Graph (RNG)
                        if max(np.linalg.norm(pi - pk), np.linalg.norm(pj - pk)) < dij:
                            is_edge = False
                            break

                    elif beta == 1:
                        # Gabriel Graph
                        if np.linalg.norm(pk - midpoint) < dij / 2:
                            is_edge = False
                            break

                    else:
                        # Общий случай
                        center1 = pi + (pj - pi) * beta / (2 * beta)
                        center2 = pj + (pi - pj) * beta / (2 * beta)
                        radius = dij / (2 * beta)

                        if np.linalg.norm(pk - center1) < radius or np.linalg.norm(pk - center2) < radius:
                            is_edge = False
                            break

                if is_edge:
                    # Добавляем оба направления
                    edge_indices.append([i, j])
                    edge_indices.append([j, i])

                    dist_tensor = torch.norm(points[i] - points[j])
                    edge_attrs.append(dist_tensor)
                    edge_attrs.append(dist_tensor)

        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long, device=points.device).t()
            edge_attr = torch.stack(edge_attrs).unsqueeze(1)  # shape: (N_edges, 1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=points.device)
            edge_attr = torch.zeros((0, 1), dtype=torch.float, device=points.device)

        return edge_index, edge_attr

    def forward(self, batch):

        reconstructed_labels = []
        reconstructed_values = []
        latent_list = []
        beta_values = [0, 1, 2]
        beta_edges = {}

        for _, obs in enumerate(batch):
            # Encode each vector to latent vector
            latent = self.encoder(obs)

            # Create a betta-skeleton graph (with beta = 1 we got the special case of Gabriel Graph)
            edge_index, edge_attr = self.create_gabriel_graph(latent)
            for beta in beta_values:
                if beta not in beta_edges:
                    beta_edges[beta] = [[],[]]
                edge_index, edge_attr = self.build_beta_skeleton(latent, beta)
                beta_edges[beta][0].append(edge_index)
                beta_edges[beta][1].append(edge_attr)

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

        return (batch[:, :, :4], batch[:, :, 4].reshape(64, 12, 1),
                torch.stack(reconstructed_labels), torch.stack(reconstructed_values),
                torch.stack(latent_list), beta_edges)


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

def train_model(model, dataloader, epochs, lr, param_schema, save_path="trained_model.pth"):
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
            true_distribution, true_values, predicted_logits, predicted_values, latent_batch, beta_edges = model(batch)

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

            idx = 11
            obs = batch[idx]
            variable_order = []

            for row in obs:
                key = tuple(row[:4].int().tolist())
                var_id = GRAY_INVERTED_MAPPING.get(key, None)
                variable_order.append(var_id)
            sorted_schema = [param_schema[i - 1] for i in variable_order]

            graphs = {}
            for key in beta_edges.keys():
                edge_index_list, edge_attr_list = beta_edges[key][0], beta_edges[key][1]
                graph = visualize_graph(
                    latent_points=latent_batch[idx],
                    edge_index=edge_index_list[idx],
                    edge_attr=edge_attr_list[idx],
                    parameters={"epoch": epoch, "beta": key},
                    param_schema=sorted_schema
                )
                graphs[key] = graph
            visualise_growth(graphs, epoch)

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

@dataclass
class NodeDescriptor:
    name: str
    group: str

group_color_map = {
    "self_vel": "#1f77b4",
    "landmark": "#2ca02c",
    "target": "#f54242",
    "agent": "#8202fa",
    "unknown": "#7f7f7f"
}

def visualize_graph(latent_points, edge_index, edge_attr, parameters: dict, save_dir="graphs", param_schema=None) -> nx.Graph:
    """
    Visualisation Graph

    :param latent_points:
    :param edge_index: (2, num_edges)
    :param epoch:
    :param save_dir:
    :param
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

    weights = edge_attr.detach().cpu().numpy()
    edge_labels = {}
    for (src, tgt), weight in zip(edges, weights):
        G.add_edge(src, tgt, weight=weight)
        edge_labels[(src, tgt)] = float(weight)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[1, 1, 1],
        height_ratios=[3, 1],
        wspace=0.3,
        hspace = 0.5
    )

    ax_graph = fig.add_subplot(gs[0, 0])
    ax_graph.axis('off')
    ax_graph.set_aspect('equal', adjustable='box')
    ax_graph.set_title(f"Latent Graph beta={parameters["beta"]} — Epoch {parameters["epoch"]}", fontsize=11)

    nx.draw_networkx_nodes(
        G,
        pos=positions,
        node_color=node_colors,
        node_size=60,
        ax=ax_graph,
        linewidths=0.8,
        edgecolors="black"
    )

    for i, (x, y) in positions.items():
        ax_graph.annotate(
            str(i),
            (x, y),
            textcoords="offset points",
            xytext=(0, -10),
            ha='center',
            fontsize=7,
            color='black'
        )

    nx.draw_networkx_edges(G, pos=positions, ax=ax_graph, edge_color="#2fe94e", width=1.2)

    # Draw MST
    nx.draw_networkx_edges(mst, pos=positions, ax=ax_graph, edge_color="#CA2171", width=1.6, style="dashed")

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

    # coords = np.array(coords)
    # vor = Voronoi(coords)
    # voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=1.5, line_alpha=0.6, point_size=0)

    legend_elements = [
        mpatches.Patch(color=color, label=group)
        for group, color in group_color_map.items() if group != "unknown"
    ]

    ax_graph.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(legend_elements),
        fontsize=8,
        frameon=False
    )

    ax_nodes = fig.add_subplot(gs[0, 1])
    ax_edges = fig.add_subplot(gs[0, 2])
    ax_nodes.axis('off')
    ax_edges.axis('off')

    node_table_data = []
    node_table_data.append(["No.", "Name", "X", "Y"])
    for i in range(len(latent_points)):
        name = param_schema[i].name if param_schema else f"node_{i}"
        x_val = latent_points[i][0].item()
        y_val = latent_points[i][1].item()
        node_table_data.append([str(i), name, f"{x_val:.3f}", f"{y_val:.3f}"])

    edge_table_data = []
    edge_table_data.append(["From", "To", "Weight"])
    for (src, tgt), weight in edge_labels.items():
        src_name = param_schema[src].name if param_schema else f"node_{src}"
        tgt_name = param_schema[tgt].name if param_schema else f"node_{tgt}"
        edge_table_data.append([src_name, tgt_name, f"{weight:.3f}"])

    def calculate_column_widths(data):
        if not data:
            return []

        num_cols = len(data[0])
        max_lens = [0] * num_cols

        for row_idx, row in enumerate(data):
            for col_idx, cell_value in enumerate(row):
                max_lens[col_idx] = max(max_lens[col_idx], len(str(cell_value)))

        total_len = sum(max_lens)
        if total_len == 0:
            return [1.0 / num_cols] * num_cols

        col_widths = [((length / total_len) * 0.95) + (0.05 / num_cols) for length in max_lens]
        sum_widths = sum(col_widths)
        col_widths = [w / sum_widths for w in col_widths]

        return col_widths

    node_col_widths = calculate_column_widths(node_table_data)
    edge_col_widths = calculate_column_widths(edge_table_data)

    table_nodes = ax_nodes.table(
        cellText=node_table_data[1:],
        colLabels=node_table_data[0],
        loc="upper center",
        cellLoc='center',
        colWidths=node_col_widths
    )
    table_nodes.auto_set_font_size(False)
    table_nodes.set_fontsize(7)

    table_edges = ax_edges.table(
        cellText=edge_table_data[1:],
        colLabels=edge_table_data[0],
        loc="upper center",
        cellLoc='center',
        colWidths=edge_col_widths
    )
    table_edges.auto_set_font_size(False)
    table_edges.set_fontsize(7)

    plt.tight_layout()
    filename = os.path.join(save_dir, f"graph-{parameters["beta"]}-epoch{parameters["epoch"] + 1:02d}.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    return G

def visualise_growth(graphs_info: dict, epoch, save_path="graphs"):
    """
    graphs_info: dict[nx.Graph], каждый dict должен содержать:
        - 'graph': networkx.Graph
        - 'label': подпись на графике (например, "β=0.5, узел=7")
        - optionally: 'color': цвет кривой
    """

    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    for key in graphs_info.keys():
        G = graphs_info[key]

        # Search Central Node
        centralities = nx.betweenness_centrality(G)
        central_node = max(centralities, key=centralities.get)
        print(central_node)

        def compute_growth_layers(G: nx.Graph, central_node: int, max_depth: int = 5):
            from collections import deque, defaultdict

            visited = set()
            queue = deque([(central_node, 0)])
            growth = defaultdict(set)

            while queue:
                node, depth = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                if depth > 0:
                    growth[depth].add(node)
                if depth < max_depth:
                    for neighbor in G.neighbors(node):
                        if neighbor not in visited:
                            queue.append((neighbor, depth + 1))

            return sorted([(k, len(v)) for k, v in growth.items()])

        growth_data = compute_growth_layers(G, central_node, max_depth=6)
        ks, ns = zip(*growth_data)
        cumulative = [sum(ns[:i + 1]) for i in range(len(ns))]

        ax.plot(ks, cumulative, marker='o', label=f"beta={key}")

    ax.set_title("Growth", fontsize=9)
    ax.set_xlabel("Distance from start")
    ax.set_ylabel("Number of states in the layer")
    ax.grid(True, linewidth=0.5, linestyle='--', alpha=0.6)
    ax.legend(fontsize=8)
    plt.tight_layout()

    filename = os.path.join(save_path, f"growth-epoch{epoch + 1:02d}.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")

    plt.close()


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

    param_schema: List[NodeDescriptor] = [
        NodeDescriptor("vel-x", "self_vel"),
        NodeDescriptor("vel-y", "self_vel"),
        NodeDescriptor("landmark-1-rel-x", "landmark"),
        NodeDescriptor("landmark-1-rel-y", "landmark"),
        NodeDescriptor("landmark-2-rel-x", "landmark"),
        NodeDescriptor("landmark-2-rel-y", "landmark"),
        NodeDescriptor("landmark-3-rel-x", "landmark"),
        NodeDescriptor("landmark-3-rel-y", "landmark"),
        NodeDescriptor("is_landmark_1_target", "target"),
        NodeDescriptor("is_landmark_2_target", "target"),
        NodeDescriptor("is_landmark_3_target", "target"),
        NodeDescriptor("agent_type", "agent"),
    ]
    # Hyperparameters for training
    train_model(
        model,
        dataloader,
        epochs=30,
        lr=0.0025,
        param_schema=param_schema,
        save_path=model_filename
    )
    
    print(f"Training completed! Model saved as {model_filename}")

if __name__ == "__main__":
    main()