import tyro
import csv
from typing import List
from collections import deque, defaultdict

from matplotlib.patches import Circle
from networkx.drawing import draw_networkx_edges
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.lines as mlines
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

from graphs_wrapper import (create_delaunay_graph,
                            create_beta_skeleton_graph, create_gabriel_graph,
                            create_anisotropic_graph, create_dal_graph, create_knn_graph, create_full_connected_graph)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@dataclass
class Config:
    # Model architecture parameters
    input_dim: int = 5                      # Input feature dimension for encoder
    output_dim: int = 3                     # Output dimension of latent embedding
    hidden_dim: int = 64                    # Hidden layer size used throughout GAT layers and MLP

    # Training hyperparameters
    epochs: int = 30                        # Total number of training epochs
    lr: float = 0.0025                      # Learning rate for optimizer
    model_save_path: str = "results/models"          # Path where trained model will be saved

    # Data generation
    num_samples: int = 1024                 # Number of synthetic samples to generate from the environment
    batch_size: int = 64                    # Batch size used in training

    # Learning rate scheduler parameters
    factor: float = 0.5                     # Factor by which the learning rate will be reduced
    patience: int = 5                       # Number of epochs with no improvement after which LR will be reduced

    # Other training parameters
    alpha: float = 0.1                      # Skip connection blending coefficient in GAT
    max_norm: float = 1.0                   # Maximum norm for gradient clipping
    gamma: float = 0.1

    # Metrics Parameters
    is_growth_from_central: bool = False    # Enable calculate growth from central user node

    # Parameters for configure results output
    visualise: bool = False                          # Enable create visualisation of epochs
    visual_save_path: str = "results/graphics"       # Path where visualisation will be saved
    save_metrics: bool = False                       # Whether to save metrics to disk
    data_save_path: str = "results/metrics"          # Path where metrics will be saved

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

@dataclass
class NodeDescriptor:
    name: str
    group: str


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

group_marker_map = {
    "self_vel": "D",
    "landmark": "s",
    "target": "^",
    "agent": "o",
    "unknown": "X"
}

algorithms = {
    'fully-connected': create_full_connected_graph,
    'beta-skeleton': lambda points: create_beta_skeleton_graph(points, beta=1.7),
    'delaunay': create_delaunay_graph,
    'gabriel': create_gabriel_graph,
    'kNN': create_knn_graph,
    'DAL': create_dal_graph
}

class GraphAutoEncoder(nn.Module):
    def __init__(self,
        input_dim=5,
        output_dim=3,
        hidden_dim=64,
        graph_fn = None
        ):
        super(GraphAutoEncoder, self).__init__()

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
        batch_size = batch.shape[0]
        return (batch[:, :, :4], batch[:, :, 4].reshape(batch_size, 12, 1),
                torch.stack(reconstructed_labels), torch.stack(reconstructed_values),
                torch.stack(latent_list), edge_index_list, edge_attr_list)


# Improved reconstruction loss with dimension-specific weighting
def reconstruction_loss(
        true_distribution: torch.Tensor,
        predicted_logits: torch.Tensor,
        true_values: torch.Tensor,
        predicted_values: torch.Tensor,
        epoch: int,
        total_epochs: int,
        gamma: float
):
    """
    Reconstruction loss
    """
    labels_kl = F.kl_div(F.log_softmax(predicted_logits, dim=-1), F.softmax(true_distribution, dim=-1), reduction='batchmean')
    values_mse = F.l1_loss(predicted_values, true_values)

    progress = epoch / total_epochs
    weight1 = 1.0 - (1-gamma) * progress
    weight2 = gamma + (1-gamma) * progress

    return  weight1 * labels_kl + weight2 * values_mse

def l1_loss(edge_attr_list):
    """
    L1 loss for the edges on the graph
    """
    return torch.norm(torch.cat(edge_attr_list), p=1)

def train_model(
        model,
        dataloader,
        name: str,
        epochs: int,
        lr: float,
        factor: float,
        patience: int,
        gamma: float,
        param_schema: list[NodeDescriptor],
        is_growth_from_central: bool = False,
        model_save_path: str ="results/models",
        model_filename: str = "model.pth",
        is_visual: bool = False,
        visual_save_path: str = "results/graphics",
        is_save: bool = False,
        data_save_path: str = "results/metrics"
):
    os.makedirs(model_save_path, exist_ok=True)
    """Train with only reconstruction loss for better convergence"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience
    )

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for _, (batch,) in enumerate(dataloader):

            # Forward pass
            true_distribution, true_values, predicted_logits, predicted_values, latent_batch, edge_index_list, edge_attr_list = model(batch)

            # Calculate the combined loss function
            loss = reconstruction_loss(
                true_distribution=true_distribution,
                predicted_logits=predicted_logits,
                true_values=true_values,
                predicted_values=predicted_values,
                epoch=epoch,
                total_epochs=epochs,
                gamma=gamma
            ) + 1e-6* l1_loss(edge_attr_list)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()


        avg_loss = epoch_loss / len(dataloader)

        print(f'Epoch [{epoch + 1}/{epochs}], Avg Loss: {avg_loss:.6f}')
        if is_save:
            save_loss_log(
                epoch=epoch+1,
                avg_loss=avg_loss,
                data_save_path=data_save_path
            )

        # Update learning rate based on validation loss
        scheduler.step(avg_loss)

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
            graph = create_graph(
                latent_points=latent_batch[idx],
                edge_index=edge_index_list[idx],
                edge_attr=edge_attr_list[idx],
                parameters={"epoch": epoch+1, "name": name},
                param_schema=sorted_schema,
                is_visual=is_visual,
                visual_save_path=visual_save_path
            )
            graphs[key] = {"graph": graph, "schema": sorted_schema}

            # calculate_growth(
            #     graphs,
            #     epoch+1,
            #     is_growth_from_central=is_growth_from_central,
            #     is_visual=is_visual,
            #     visual_save_path=visual_save_path,
            #     is_save=is_save,
            #     data_save_path=data_save_path,
            # )
            # calculate_graphs_metrics(graphs, epoch)

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
    }, f"{model_save_path}/{model_filename}")
    
    print(f"Model saved to {model_save_path}")


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

def create_graph(
        latent_points,
        edge_index,
        edge_attr,
        parameters: dict,
        param_schema=List[NodeDescriptor],
        is_visual: bool = False,
        visual_save_path: str = "results/graphics",
) -> nx.Graph:
    G = nx.Graph()
    positions = {}

    for i in range(latent_points.shape[0]):
        x, y = latent_points[i][0].item(), latent_points[i][1].item()
        G.add_node(i)
        positions[i] = (x, y)

    edges = edge_index.t().cpu().numpy()

    if edge_index.numel() > 0:
        if edge_attr is not None:
            weights = edge_attr.detach().cpu().numpy()
            for (src, tgt), weight in zip(edges, weights):
                G.add_edge(src, tgt, weight=weight)
        else:
            for src, tgt in edges:
                G.add_edge(src, tgt)

    if is_visual:
        weights = edge_attr.detach().cpu().numpy()
        edge_labels = {}
        for (src, tgt), weight in zip(edges, weights):
            G.add_edge(src, tgt, weight=weight)
            edge_labels[(src, tgt)] = float(weight)

        visualize_graph(
            G=G,
            positions=positions,
            latent_points=latent_points,
            edge_labels=edge_labels,
            parameters=parameters,
            param_schema=param_schema,
            picture_name=f"graph-{parameters["name"]}-epoch{parameters["epoch"] + 1:02d}.png",
            visual_save_path=visual_save_path
        )
    return G

def save_loss_log(
        epoch: int,
        avg_loss: float,
        data_save_path: str = "results/metrics"
):
    os.makedirs(data_save_path, exist_ok=True)
    filepath=f"{data_save_path}/loss_log.csv"
    file_exists = os.path.exists(filepath)

    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Epoch", "AvgLoss"])
        writer.writerow([epoch, avg_loss])

def visualize_graph(
        G,
        positions,
        latent_points,
        edge_labels,
        parameters: dict,
        param_schema=List[NodeDescriptor],
        picture_name: str = "graph.png",
        visual_save_path: str = "results/graphics",
):

    os.makedirs(visual_save_path, exist_ok=True)

    fig, ax_graph = plt.subplots()
    ax_graph.axis('off')
    ax_graph.set_aspect('equal', adjustable='box')

    nx.draw_networkx_edges(
        G, pos=positions, ax=ax_graph, edge_color="#888888", width=1.2
    )

    # Draw MST
    mst = nx.minimum_spanning_tree(G)
    nx.draw_networkx_edges(
        mst, pos=positions, ax=ax_graph, edge_color="#000000", width=1.6, style="dashed"
    )

    for group, marker in group_marker_map.items():
        indices = [i for i, node in enumerate(param_schema) if getattr(node, 'group', 'unknown') == group]
        if not indices:
            continue
        xs = [positions[i][0] for i in indices]
        ys = [positions[i][1] for i in indices]
        ax_graph.scatter(
            xs, ys,
            marker=marker,
            s=100,
            c="#222222",
            edgecolors="#0a0a0a",
            label=group,
            zorder=3
        )
        # Подписи номеров
        for idx in indices:
            ax_graph.annotate(
                str(idx),
                positions[idx],
                textcoords="offset points",
                xytext=(0, -20),
                ha='center',
                fontsize=7,
                color='black'
            )

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

    coords = []
    for i in range(latent_points.shape[0]):
        x, y = latent_points[i][0].item(), latent_points[i][1].item()
        coords.append([x, y])
    coords = np.array(coords)
    vor = Voronoi(coords)
    voronoi_plot_2d(vor, ax=ax_graph, show_vertices=False, line_colors='gray', line_width=1.5, line_alpha=0.6, point_size=0)

    legend_elements = [
        mlines.Line2D(
            [], [], color='#0a0a0a',
            marker=marker,
            linestyle='None',
            markersize=8,
            label=group
        )
        for group, marker in group_marker_map.items()
    ]

    ax_graph.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(legend_elements),
        fontsize=8,
        frameon=False
    )

    node_table_keys = ["No.", "Name", "X", "Y"]
    node_table_data = []
    for i in range(len(latent_points)):
        data = {}
        data["No."] = i
        data["Name"] = param_schema[i].name if param_schema else f"node_{i}"
        data["X"] = f"{latent_points[i][0].item():.3f}"
        data["Y"] = f"{latent_points[i][1].item():.3f}"
        node_table_data.append(data)
    save_data_to_csv(node_table_keys, node_table_data, f"nodes-{picture_name}.csv")

    edge_table_keys = ["From", "To", "Weight"]
    edge_table_data = []
    for (src, tgt), weight in edge_labels.items():
        data = {}
        data["From"] = param_schema[src].name if param_schema else f"node_{src}"
        data["To"] = param_schema[tgt].name if param_schema else f"node_{tgt}"
        data["Weight"] = f"{weight:.3f}"
        edge_table_data.append(data)
    save_data_to_csv(edge_table_keys, edge_table_data, f"edges-{picture_name}.csv")

    plt.tight_layout()
    filename = os.path.join(visual_save_path, picture_name)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

def calculate_growth(
        graphs_info: dict,
        epoch: int,
        is_growth_from_central: bool = False,
        is_visual: bool = False,
        visual_save_path: str = "results/graphics",
        is_save: bool = False,
        data_save_path: str = "results/metrics"
):
    all_growth_records = []

    for key, graph_info in graphs_info.items():
        G = graph_info["graph"]
        schema = graph_info["schema"]
        start_node = 0

        if is_growth_from_central:
            # Search Central Node
            centralities = nx.betweenness_centrality(G)
            start_node = max(centralities, key=centralities.get)
            print(f"Central node for {key}: {start_node}")
        else:
            for idx in  range(0,len(schema)):
                if schema[idx].name == "agent_type":
                    start_node = idx
                    break

        growth_data = compute_growth_layers(G, start_node, max_depth=len(schema))

        if not growth_data:
            print(f"[Epoch {epoch}] ⚠ growth_data is empty for {key}. Skipping.")
            continue

        ks, ns = zip(*growth_data)
        cumulative = [sum(ns[:i + 1]) for i in range(len(ns))]

        # Добавление данных к общему списку
        for i in range(len(ks)):
            all_growth_records.append({
                "GraphKey": key,
                "Layer": ks[i],
                "NewNodes": ns[i],
                "CumulativeNodes": cumulative[i]
            })

        if is_visual:
            plot_path = os.path.join(visual_save_path, f"growth-epoch{epoch:02d}.png")
            plot_growth_curves(all_growth_records, plot_path)

        if is_save:
            csv_path = os.path.join(data_save_path, f"growth_all_data-epoch{epoch:02d}.csv")
            save_growth_data_csv(all_growth_records, csv_path)

def compute_growth_layers(graph: nx.Graph, start_node: int, max_depth: int = 5):
    visited = set()
    queue = deque([(start_node, 0)])
    growth = defaultdict(set)
    growth[0].add(start_node)

    while queue:
        node, depth = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        if depth > 0:
            growth[depth].add(node)
        if depth < max_depth:
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))

    return sorted([(k, len(v)) for k, v in growth.items()])

def save_data_to_csv(keys: List[str], data: List[dict], filename: str, legend_save_path="results/graphics/"):
    os.makedirs(os.path.dirname(legend_save_path), exist_ok=True)
    filepath = f"{legend_save_path}/{filename}"
    file_exists = os.path.exists(filepath)

    with open(filepath, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)

def save_growth_data_csv(data: List[dict], filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["GraphKey", "Layer", "NewNodes", "CumulativeNodes"])
        writer.writeheader()
        writer.writerows(data)

def plot_growth_curves(growth_records: List[dict], output_path: str):
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    grouped = defaultdict(list)
    for row in growth_records:
        grouped[row["GraphKey"]].append(row)

    for key, records in grouped.items():
        ks = [r["Layer"] for r in records]
        cumulative = [r["CumulativeNodes"] for r in records]
        ax.plot(ks, cumulative, marker='o', label=f"{key}")

    ax.set_title("Growth", fontsize=9)
    ax.set_xlabel("Distance from start")
    ax.set_ylabel("Number of states in the layer")
    ax.grid(True, linewidth=0.5, linestyle='--', alpha=0.6)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def calculate_graph_metrics(G):
    metrics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "avg_clustering": nx.average_clustering(G), "assortativity": nx.degree_assortativity_coefficient(G),
        "num_components": nx.number_connected_components(G)
    }

    largest_cc = max(nx.connected_components(G), key=len)
    metrics["size_largest_cc"] = len(largest_cc)

    if nx.is_connected(G):
        metrics["avg_path_length"] = nx.average_shortest_path_length(G)
        metrics["diameter"] = nx.diameter(G)
        metrics["radius"] = nx.radius(G)
    else:
        metrics["avg_path_length"] = None
        metrics["diameter"] = None
        metrics["radius"] = None

    return metrics

def calculate_graphs_metrics(
        graphs_info: dict,
        epoch: int,
        is_save: bool = False,
        data_save_path: str = "results/metrics"
):
    rows = []
    for beta, info in graphs_info.items():
        G = info['graph']
        metrics = calculate_graph_metrics(G)
        metrics["epoch"] = epoch
        metrics["name"] = beta
        rows.append(metrics)

    if is_save:
        save_graphs_metrics(
            metrics=rows,
            metrics_save_path=data_save_path
        )

def save_graphs_metrics(
        metrics: list[dict],
        metrics_save_path="results/metrics"
):
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    filepath = f"{metrics_save_path}/graph_metrics_all.csv"
    file_exists = os.path.exists(filepath)
    fieldnames = ["epoch", "name"] + [k for k in metrics[0] if k not in ["epoch", "name", "label"]]

    with open(filepath, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(metrics)

# Main function
def main():
    config = tyro.cli(Config)
    # Generate dataset
    dataloader = generate_sample_data(
        num_samples=config.num_samples,
        batch_size=config.batch_size,
    )



    for algo_name, graph_fn in algorithms.items():
        print(f"\nTesting algorithm: {algo_name}")
        # Create model
        model = GraphAutoEncoder(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
            graph_fn=graph_fn,
        ).to(device)
        print(model)

        # Create a timestamped model filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_filename = f"{algo_name}_{timestamp}.pth"

        train_model(
            model,
            dataloader,
            name=algo_name,
            epochs=config.epochs,
            lr=config.lr,
            factor=config.factor,
            patience=config.patience,
            gamma=config.gamma,
            param_schema=param_schema,
            is_growth_from_central=config.is_growth_from_central,
            model_save_path=config.model_save_path,
            model_filename=model_filename,
            is_visual=config.visualise,
            visual_save_path=config.visual_save_path,
            is_save=config.save_metrics,
            data_save_path=config.data_save_path,
        )
    
        print(f"Training completed! Model saved as {model_filename}")

if __name__ == "__main__":
    main()