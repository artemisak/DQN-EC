import os

import torch

from listener_gae import GraphAutoEncoder
from graphs_wrapper import create_delaunay_graph
from dataset import SyntheticData
from hypergraph_geometric import PyGHypergraphBuilder, visualize_hypergraph, analyze_hypergraph_connectivity

def shift_graph(data, dx=0.0, dy=0.0, inplace=True):
    """Translate 2D node positions by (dx, dy)."""
    if not inplace:
        data = data.clone()

    if data.pos is None:
        raise ValueError("data.pos is None")

    offset = torch.tensor([dx, dy], dtype=data.pos.dtype, device=data.pos.device)
    data.pos = data.pos + offset
    return data

def save_hypergraph(hg, path):
    """Saves a hypergraph object to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(hg, path)
    print(f"Hypergraph saved to {path}")

def load_hypergraph(path):
    """Loads a hypergraph object from a file."""
    if os.path.exists(path):
        hg = torch.load(path)
        print(f"Hypergraph loaded from {path}")
        return hg
    else:
        print(f"File not found: {path}")
        return None

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
    # Add saving dataset to ".pt"

    idx = 0

    with torch.no_grad():
        _, _, sp_graphs = sp_model(generator.speaker[idx].unsqueeze(0).to(device))
        _, _, ls_graphs = ls_model(generator.listener[idx].unsqueeze(0).to(device))

        components = []

        sp_graph = sp_graphs[0]
        g1 = shift_graph(sp_graph, dx=-5, dy=0.0)
        components.append(g1)

        ls_graph = ls_graphs[0]
        g2 = shift_graph(ls_graph, dx=+5, dy=0.0)
        components.append(g2)
        print(components)

        # Hypergraph
        # Build strict hypergraph
        print("\n2. Building STRICT hypergraph...")
        hg_strict_path = "data/hypergraph_strict.pt"
        hg_strict = PyGHypergraphBuilder.build_strict(
            data_list=components,
            s=3,
            hub_component=1
        )
        save_hypergraph(hg_strict, hg_strict_path)

        print(f"   Total nodes: {hg_strict.num_nodes}")
        print(f"   Total edges: {hg_strict.num_edges}")
        print(f"   Hyperedges created: {len(hg_strict.hyperedges)}")
        print(hg_strict)
        # Analyze connectivity
        analysis_strict = analyze_hypergraph_connectivity(hg_strict)
        print(f"   Connected: {analysis_strict['is_connected']}")
        print(f"   Average hyperedge size: {analysis_strict['avg_hyperedge_size']:.2f}")
        print(f"   Average node degree: {analysis_strict['avg_degree']:.2f}")

        # Build relaxed hypergraph (no constraints)
        print("\n3. Building RELAXED hypergraph (no constraints)...")
        hg_relaxed_path = "data/hypergraph_relaxed.pt"
        hg_relaxed = PyGHypergraphBuilder.build_relaxed(
            data_list=components,
            s=3,
            rho=None
        )
        save_hypergraph(hg_relaxed, hg_relaxed_path)

        print(f"   Total nodes: {hg_relaxed.num_nodes}")
        print(f"   Total edges: {hg_relaxed.num_edges}")
        print(f"   Hyperedges created: {len(hg_relaxed.hyperedges)}")
        print(hg_relaxed)
        # Analyze connectivity
        analysis_relaxed = analyze_hypergraph_connectivity(hg_relaxed)
        print(f"   Connected: {analysis_relaxed['is_connected']}")
        print(f"   Average hyperedge size: {analysis_relaxed['avg_hyperedge_size']:.2f}")
        print(f"   Average node degree: {analysis_relaxed['avg_degree']:.2f}")

        # Build relaxed hypergraph (with radius constraint)
        print("\n4. Building RELAXED hypergraph (with radius constraint)...")
        hg_constrained_path = "data/hypergraph_constrained.pt"
        hg_constrained = PyGHypergraphBuilder.build_relaxed(
            data_list=components,
            s=4,
            rho=0.5,
            force_connect=True
        )
        save_hypergraph(hg_constrained, hg_constrained_path)

        print(f"   Total nodes: {hg_constrained.num_nodes}")
        print(f"   Total edges: {hg_constrained.num_edges}")
        print(f"   Hyperedges created: {len(hg_constrained.hyperedges)}")
        print(hg_constrained)
        # Analyze connectivity
        analysis_constrained = analyze_hypergraph_connectivity(hg_constrained)
        print(f"   Connected: {analysis_constrained['is_connected']}")
        print(f"   Average hyperedge size: {analysis_constrained['avg_hyperedge_size']:.2f}")
        print(f"   Average node degree: {analysis_constrained['avg_degree']:.2f}")

        # Show feature preservation
        print("\n5. Verifying feature preservation...")
        print(f"   Node features shape: {hg_strict.x.shape}")
        print(f"   Node positions shape: {hg_strict.pos.shape}")
        print(f"   Edge attributes shape: {hg_strict.edge_attr.shape if hg_strict.edge_attr is not None else 'None'}")
        print(f"   Component labels shape: {hg_strict.comp.shape}")

        # Show edge types
        if hasattr(hg_strict, 'edge_type'):
            n_original = (hg_strict.edge_type == 0).sum().item()
            n_hyperedge = (hg_strict.edge_type == 1).sum().item()
            print(f"   Original edges: {n_original}")
            print(f"   Hyperedge connections: {n_hyperedge}")

        # Visualize the hypergraphs
        print("\n6. Generating visualizations...")

        visualize_hypergraph(
            hg_strict,
            title="Strict Hypergraph",
            save_path="hypergraph_strict_pyg.png",
            show_hyperedge_hulls=True
        )

        visualize_hypergraph(
            hg_relaxed,
            title="Relaxed Hypergraph (No Constraints)",
            save_path="hypergraph_relaxed_pyg.png",
            show_hyperedge_hulls=True
        )

        visualize_hypergraph(
            hg_constrained,
            title="Relaxed Hypergraph (ρ=2.0)",
            save_path="hypergraph_constrained_pyg.png",
            show_hyperedge_hulls=True
        )

        # Print connectivity comparison
        print("\n7. Connectivity Comparison:")
        print("   " + "-" * 50)
        print(f"   {'Metric':<30} {'Strict':<10} {'Relaxed':<10} {'Constrained':<10}")
        print("   " + "-" * 50)
        print(
            f"   {'Connected':<30} {str(analysis_strict['is_connected']):<10} {str(analysis_relaxed['is_connected']):<10} {str(analysis_constrained['is_connected']):<10}")
        print(
            f"   {'Hyperedges':<30} {analysis_strict['n_hyperedges']:<10} {analysis_relaxed['n_hyperedges']:<10} {analysis_constrained['n_hyperedges']:<10}")
        print(
            f"   {'Avg Hyperedge Size':<30} {analysis_strict['avg_hyperedge_size']:<10.2f} {analysis_relaxed['avg_hyperedge_size']:<10.2f} {analysis_constrained['avg_hyperedge_size']:<10.2f}")
        print(
            f"   {'Avg Node Degree':<30} {analysis_strict['avg_degree']:<10.2f} {analysis_relaxed['avg_degree']:<10.2f} {analysis_constrained['avg_degree']:<10.2f}")
        print(
            f"   {'Max Node Degree':<30} {analysis_strict['max_degree']:<10} {analysis_relaxed['max_degree']:<10} {analysis_constrained['max_degree']:<10}")