import torch
import networkx as nx
import matplotlib.pyplot as plt

from listener_gae import GraphAutoEncoder
from graphs_wrapper import create_delaunay_graph
from dataset import SyntheticData


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphAutoEncoder(
        input_dim=5,
        output_dim=3,
        hidden_dim=128,
        graph_fn=create_delaunay_graph
    ).to(device)
    state = torch.load('listener_gae.pt', map_location=device)
    model.load_state_dict(state)
    model.eval()

    generator = SyntheticData()

    idx = 0
    sample = generator.listener[idx]
    sample = sample.unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, graphs = model(sample)

    graph = graphs[0]

    edge_index = graph.edge_index.cpu()
    pos_tensor = graph.pos.cpu()
    edge_attr = None
    if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
        edge_attr = graph.edge_attr.squeeze(-1).cpu()

    G = nx.Graph()
    num_nodes = pos_tensor.shape[0]
    G.add_nodes_from(range(num_nodes))

    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    if edge_attr is not None:
        weights = edge_attr.tolist()
        for u, v, w in zip(src, dst, weights):
            G.add_edge(u, v, weight=float(w))
    else:
        for u, v in zip(src, dst):
            G.add_edge(u, v)

    pos = {i: (float(pos_tensor[i, 0]), float(pos_tensor[i, 1])) for i in range(num_nodes)}

    widths = [G[u][v]['weight'] for u, v in G.edges()]

    plt.figure(figsize=(6, 6))
    nx.draw(
        G,
        pos=pos,
        width=widths
    )
    plt.title("Sampled Graph")
    plt.show()
