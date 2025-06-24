from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from matplotlib.patches import Circle
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.patches as mpatches
from dataclasses import dataclass
from pettingzoo.mpe import simple_speaker_listener_v4

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import os
from datetime import datetime

@dataclass
class NodeDescriptor:
    name: str       # Например: "self_vel-x"
    group: str      # Тип: "self_vel", "landmark"

group_color_map = {
    "self_vel": "#1f77b4",
    "landmark": "#2ca02c",
    "target": "#f54242",
    "agent": "#8202fa",
    "unknown": "#7f7f7f"
}

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

        self.alpha = 0.1 # TODO: make it learnable
        # self.alhpa = nn.Sigmoid()

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

    def beta_skeleton_graph(self, points, beta=1):
        """
            Построение beta-skeleton графа на PyTorch тензорах.

            :param points: тензор shape (N, 2) или (N, D) с координатами точек
            :param beta: параметр beta (1 - Gabriel graph)
            :return: список рёбер [[i, j], ...] в виде Python списка (или можно вернуть тензор)
            """
        n = points.shape[0]
        edges = []

        for i in range(n):
            p1 = points[i]
            for j in range(i + 1, n):
                p2 = points[j]
                d = torch.norm(p1 - p2)

                if d < 1e-10:
                    continue

                is_edge = True

                if abs(beta - 1.0) < 1e-8:
                    # Gabriel graph: проверяем, нет ли точки внутри окружности с диаметром p1-p2
                    center = (p1 + p2) / 2
                    radius = d / 2

                    for k in range(n):
                        if k == i or k == j:
                            continue
                        pk = points[k]
                        dist_to_center = torch.norm(pk - center)
                        if dist_to_center < radius:
                            is_edge = False
                            break
                else:
                    if beta < 1e-5:
                        continue

                    center = (p1 + p2) / 2
                    direction = (p2 - p1) / d
                    perpendicular = torch.tensor([-direction[1], direction[0]], device=points.device)

                    offset = 0.0
                    if beta > 1:
                        offset = d * torch.sqrt(1 / (4 * beta ** 2) - 1 / 4)

                    if beta > 1:
                        c1 = center + offset * perpendicular
                        c2 = center - offset * perpendicular
                        radius = d / (2 * beta)

                        for k in range(n):
                            if k == i or k == j:
                                continue
                            pk = points[k]
                            if (torch.norm(pk - c1) < radius) or (torch.norm(pk - c2) < radius):
                                is_edge = False
                                break
                    else:  # beta < 1
                        radius = d / (2 * beta)
                        for k in range(n):
                            if k == i or k == j:
                                continue
                            pk = points[k]
                            if torch.norm(pk - center) > radius:
                                is_edge = False
                                break

                if is_edge:
                    edges.append([i, j])
                    edges.append([j, i])  # двунаправленное ребро

        return edges

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
            # graph_latent_np = latent.detach().cpu().numpy()
            edges = self.beta_skeleton_graph(latent)
            edge_index = torch.tensor(edges, dtype=torch.long, device=latent.device).t()
            edge_attr = torch.norm(latent[edge_index[0]] - latent[edge_index[1]], dim=1, keepdim=True)

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


# Improved reconstruction loss with dimension-specific weighting
def reconstruction_loss(original, reconstructed):
    """
    Reconstruction loss
    """
    return F.mse_loss(reconstructed, original)


def l1_loss(edge_attr_list):
    """
    L1 loss for the edges on the graph
    """
    return torch.norm(torch.cat(edge_attr_list), p=1)

def train_model(model, dataloader, epochs, lr, param_schema, save_path="model"):
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
            original, reconstructed, latent_batch, edge_index_list, edge_attr_list = model(batch)

            # Calculate the combined loss function
            loss = reconstruction_loss(original, reconstructed) + 1e-6  * l1_loss(edge_attr_list)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 30 == 0:
            print(f'Epoch: {epoch}')
            print('='*50, 'Original', '='*50)
            print(torch.round(original[0], decimals=1))
            print('='*50, 'Reconstructed', '='*50)
            print(torch.round(reconstructed[0], decimals=1))
            visualize_and_save_gabriel_graph(
                latent_points=latent_batch[0].detach().cpu(),
                edge_index=edge_index_list[0].detach().cpu(),
                attn_weights=edge_attr_list[0].detach().cpu(),
                epoch=epoch,
                param_schema=param_schema
            )

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
def generate_sample_data(num_samples=256, batch_size=32):
    """Sample the data from the environment with a random policy"""

    env = simple_speaker_listener_v4.parallel_env(max_cycles=num_samples)
    env.reset()
    observations = []

    def create_observation_pair(message):
        """Create both observation variants for a message"""

        # First observation (agent_type=0)
        obs1 = torch.zeros(12, 12)
        obs1[:, :11] = torch.eye(12, 11)
        obs1[0:8, 11] = torch.tensor(message[:8])  # velocity and landmarks
        obs1[8:11, 11] = -1  # masked out
        obs1[11, 11] = 0  # agent_type

        # Second observation (agent_type=1)
        obs2 = torch.zeros(12, 12)
        obs2[:, :11] = torch.eye(12, 11)
        obs2[0:8, 11] = -1  # masked out
        obs2[8:11, 11] = torch.tensor(message[8:])  # target flags
        obs2[11, 11] = 1  # agent_type

        return [obs1, obs2]

    while env.agents and len(observations) < num_samples:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, *_ = env.step(actions)
        observations.extend(create_observation_pair(obs['listener_0']))

    env.close()

    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.stack(observations).to(device)),
        batch_size=batch_size,
        shuffle=True
    )

def visualize_and_save_gabriel_graph(latent_points, edge_index, attn_weights, epoch, save_dir="graphs", param_schema=None):
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

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(
        nrows=1,  # Одна строка
        ncols=3,  # Три колонки
        width_ratios=[1, 1, 1],  # Соотношение ширины: граф, таблица узлов, таблица рёбер
        wspace=0.3  # Горизонтальный отступ
    )

    ax_graph = fig.add_subplot(gs[0, 0])  # Граф в первой колонке
    ax_nodes = fig.add_subplot(gs[0, 1])  # Таблица узлов во второй колонке
    ax_edges = fig.add_subplot(gs[0, 2])  # Таблица рёбер в третьей колонке

    ax_graph.axis('off')
    ax_graph.set_aspect('equal', adjustable='box')
    ax_graph.set_title(f"Latent Graph — Epoch {epoch + 1}", fontsize=11)
    ax_nodes.axis('off')
    ax_edges.axis('off')
    # Рисуем ноды
    nx.draw_networkx_nodes(
        G,
        pos=positions,
        node_color=node_colors,
        node_size=60,
        ax=ax_graph,
        linewidths=0.8,
        edgecolors="black"
    )

    # Подписи над точками — выше и правее, мелко, без фона
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

    # vor = Voronoi(coords)
    # voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=1.5, line_alpha=0.6, point_size=0)

    # Сначала создаём стандартные элементы легенды по группам
    legend_elements = [
        mpatches.Patch(color=color, label=group)
        for group, color in group_color_map.items() if group != "unknown"
    ]

    ax_graph.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.08),  # под графиком
        ncol=len(legend_elements),
        fontsize=8,
        frameon=False
    )

    # Определяем ширину имени для визуального контроля (можно не обязательно)
    if param_schema:
        max_name_len = max(len(p.name) for p in param_schema)
    else:
        max_name_len = max(len(f"node_{i}") for i in range(len(latent_points)))

    # Формируем данные для таблицы узлов (заголовок + строки)
    node_table_data = []
    node_table_data.append(["No.", "Name", "X", "Y"])
    for i in range(len(latent_points)):
        name = param_schema[i].name if param_schema else f"node_{i}"
        x_val = latent_points[i][0].item()
        y_val = latent_points[i][1].item()
        node_table_data.append([str(i), name, f"{x_val:.3f}", f"{y_val:.3f}"])

    # Формируем данные для таблицы рёбер (заголовок + строки)
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

        # Создаем таблицы внутри ax_table
    table_nodes = ax_nodes.table(
        cellText=node_table_data[1:],
        colLabels=node_table_data[0],
        loc="upper center",
        cellLoc='center',
        colWidths=node_col_widths
    )
    table_nodes.auto_set_font_size(False)
    table_nodes.set_fontsize(7)

    # Таблица рёбер
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
    filename = os.path.join(save_dir, f"epoch{epoch + 1:02d}.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


# Main function
def main():
    # Create directories for saving outputs
    os.makedirs("training_graphs", exist_ok=True)

    # Generate dataset
    dataloader = generate_sample_data()

    # Create model
    model = ImprovedGraphAutoEncoder().to(device)
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
        epochs=300,
        lr=0.0025,
        param_schema=param_schema,
        save_path=model_filename,
    )

    print(f"Training completed! Model saved as {model_filename}")

if __name__ == "__main__":
    main()