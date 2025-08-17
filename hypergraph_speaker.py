import torch

from listener_gae import GraphAutoEncoder
from graphs_wrapper import create_delaunay_graph
from dataset import SyntheticData
from hypergraph_geometric import PyGHypergraphBuilder, visualize_hypergraph

def shift_graph(data, dx=0.0, dy=0.0, inplace=True):
    """Translate 2D node positions by (dx, dy)."""
    if not inplace:
        data = data.clone()

    if data.pos is None:
        raise ValueError("data.pos is None")

    offset = torch.tensor([dx, dy], dtype=data.pos.dtype, device=data.pos.device)
    data.pos = data.pos + offset
    return data

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ls_model = GraphAutoEncoder(
        input_dim=5,
        output_dim=3,
        hidden_dim=128,
        graph_fn=create_delaunay_graph
    ).to(device)
    state = torch.load('listener_gae.pt', map_location=device)
    ls_model.load_state_dict(state)
    ls_model.eval()

    sp_model = GraphAutoEncoder(
        input_dim=772,
        output_dim=3,
        hidden_dim=128,
        graph_fn=create_delaunay_graph
    ).to(device)
    state = torch.load('speaker_gae.pt', map_location=device)
    sp_model.load_state_dict(state)
    sp_model.eval()

    generator = SyntheticData()

    idx = 0

    with torch.no_grad():
        _, _, sp_graphs = sp_model(generator.speaker[idx].unsqueeze(0).to(device))
        _, _, ls_graphs = ls_model(generator.listener[idx].unsqueeze(0).to(device))

        # Normalize
        # absolute positions (world coordinates)
        # speaker_pos  = env.unwrapped.world.agents[0].state.p_pos  # speaker_0
        # listener_pos = env.unwrapped.world.agents[1].state.p_pos  # listener_0

        components = []

        sp_graph = sp_graphs[0]
        sp_graph.edge_attr = sp_graph.edge_attr.unsqueeze(-1)
        g1 = shift_graph(sp_graph, dx=-5, dy=0.0)
        components.append(g1)

        ls_graph = ls_graphs[0]
        ls_graph.edge_attr = ls_graph.edge_attr.unsqueeze(-1)
        g2 = shift_graph(ls_graph, dx=+5, dy=0.0)
        components.append(g2)
        print(components)
        # Hypergraph
        # Build strict hypergraph
        print("\n2. Building STRICT hypergraph...")
        hg_strict = PyGHypergraphBuilder.build_strict(
            data_list=components,
            s=3,
            hub_component=1
        )
        print(f"   Total nodes: {hg_strict.num_nodes}")
        print(f"   Total edges: {hg_strict.num_edges}")
        print(f"   Hyperedges created: {len(hg_strict.hyperedges)}")
        print(hg_strict)
        print("\n6. Generating visualizations...")

        visualize_hypergraph(
            hg_strict,
            title="Strict Hypergraph",
            save_path="hypergraph_strict_pyg.png",
            show_hyperedge_hulls=True
        )