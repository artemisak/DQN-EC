import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

from graphs_wrapper import create_delaunay_graph

from dataset import SyntheticData


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class GraphAutoEncoder(nn.Module):

    def __init__(self, input_dim=5, output_dim=3, hidden_dim=128, graph_fn=None):
        super(GraphAutoEncoder, self).__init__()

        self.graph_fn = graph_fn if graph_fn is not None else create_delaunay_graph

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

            graph = Data(x=latent[:, 2].reshape(-1, 1), edge_index=edge_index, edge_attr=edge_attr, pos=latent[:, :2])
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

def reconstruction_loss(true_distribution, predicted_logits, true_values, predicted_values, w1, w2):
    labels_kl = F.kl_div(F.log_softmax(predicted_logits, dim=-1),
                         F.softmax(true_distribution, dim=-1),
                         reduction='batchmean')
    values_mae = F.l1_loss(predicted_values.squeeze(-1), true_values)

    first_term = w1 * labels_kl
    second_term = w2 * values_mae
    total = first_term + second_term

    return first_term, second_term, total

def regularization_loss(edge_attr_list):
    return torch.norm(torch.cat(edge_attr_list), p=1)

def train(model, generator, epochs, lr):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    track_epochs = [1, 5, 10, 15, 20, 25, 30]
    metrics = {epoch: {} for epoch in track_epochs}

    for epoch in range(epochs):
        model.train()
        epoch_metrics = {
            'first_term': [],
            'second_term': [],
            'graph_loss': [],
            'total_loss': []
        }

        for batch_idx, (batch,) in enumerate(generator.loader):

            predicted_logits, predicted_values, graphs = model(generator.listener[batch])

            progress = epoch / epochs
            w1 = 1.0 - 0.9 * progress
            w2 = 0.1 + 0.9 * progress

            first_term, second_term, recon_loss = reconstruction_loss(
                generator.listener[batch][:, :, :4], predicted_logits,
                generator.listener[batch][:, :, 4], predicted_values,
                w1, w2
            )

            graph_loss_val = 1e-6 * regularization_loss([graph.edge_attr for graph in graphs])
            total_loss = recon_loss + graph_loss_val

            epoch_metrics['first_term'].append(first_term.item())
            epoch_metrics['second_term'].append(second_term.item())
            epoch_metrics['graph_loss'].append(graph_loss_val.item())
            epoch_metrics['total_loss'].append(total_loss.item())

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}

        if epoch + 1 in track_epochs:
            metrics[epoch + 1] = avg_metrics
            print(f"Epoch {epoch + 1}: Total Loss = {avg_metrics['total_loss']:.6f}")

        scheduler.step(avg_metrics['total_loss'])

    return model

if __name__ == "__main__":

    print("Training...")

    generator = SyntheticData()

    model = train(GraphAutoEncoder(input_dim=5,
                                   output_dim=3,
                                   hidden_dim=128,
                                   graph_fn=create_delaunay_graph).to(device),
                  generator=generator, epochs=30, lr=0.0025)

    torch.save(model.state_dict(), "listener_gae.pt")

    print('Done!')