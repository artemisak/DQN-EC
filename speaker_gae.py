import numpy as np
import torch

from dataset import SyntheticData
from listener_gae import GraphAutoEncoder, reconstruction_loss, regularization_loss
from graphs_wrapper import create_delaunay_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

            predicted_logits, predicted_values, graphs = model(generator.speaker[batch])

            progress = epoch / epochs
            w1 = 1.0 - 0.9 * progress
            w2 = 0.1 + 0.9 * progress

            first_term, second_term, recon_loss = reconstruction_loss(
                generator.speaker[batch][:, :, :4], predicted_logits,
                generator.speaker[batch][:, :, 4], predicted_values,
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

    model = train(
        GraphAutoEncoder(
            input_dim=772,
            output_dim=3,
            hidden_dim=128,
            graph_fn=create_delaunay_graph
        ).to(device),
        generator=generator,
        epochs=30,
        lr=0.0025
    )

    torch.save(model.state_dict(), "speaker_gae.pt")

    print('Done!')