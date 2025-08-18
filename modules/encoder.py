import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

class GraphAutoEncoder(nn.Module):

    def __init__(self, input_dim=5, output_dim=3, hidden_dim=128, graph_fn=None):
        super(GraphAutoEncoder, self).__init__()

        self.graph_fn = graph_fn

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.encoder.apply(self.kaiming_init)

        self.gcn1 = GATv2Conv(in_channels=1, out_channels=hidden_dim, edge_dim=1)
        self.gcn2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=1)

        self.gcn3 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.label_head = nn.Linear(hidden_dim, 4)

        self.gcn4 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.skip_connection = nn.Linear(output_dim, hidden_dim)

        self.alpha = 0.1

    def kaiming_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def forward(self, batch):
        graphs = []
        reconstructed_labels = []
        reconstructed_values = []

        for _, obs in enumerate(batch):

            latent = self.encoder(obs)

            edge_index, edge_attr = self.graph_fn(latent[:, :2])

            graph = Data(x=latent[:, 2].reshape(-1, 1),
                         edge_index=edge_index,
                         edge_attr=edge_attr.reshape(-1, 1),
                         pos=latent[:, :2])
            graphs.append(graph)

            x1 = F.relu(self.gcn1(graph.x, graph.edge_index, edge_attr=graph.edge_attr))
            x2 = F.relu(self.gcn2(x1, graph.edge_index, edge_attr=graph.edge_attr))

            x3 = F.relu(self.gcn3(x2, graph.edge_index) + self.alpha * self.skip_connection(latent))
            logits = self.label_head(x3)
            reconstructed_labels.append(logits)

            x4 = F.relu(self.gcn4(x2, graph.edge_index) + self.alpha * self.skip_connection(latent))
            values = self.value_head(x4)
            reconstructed_values.append(values)

        return (torch.stack(reconstructed_labels),
                torch.stack(reconstructed_values),
                graphs)