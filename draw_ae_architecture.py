from graphviz import Digraph


def draw_autoencoder_architecture(output_path="graph_autoencoder_architecture", dpi=300):
    # Initialize graph
    dot = Digraph("Graph AutoEncoder Architecture", format="png")
    dot.attr(fontname="Helvetica", bgcolor="white")
    # Minimize vertical spacing and edge lengths for compact layout
    dot.graph_attr.update(
        size="8,8",
        ratio="1",
        rankdir="TB",
        nodesep="0.1",      # horizontal node separation
        ranksep="0.1",      # vertical rank separation reduced
        splines="line",     # straight lines for shorter edges
        center="true",
        dpi=str(dpi)
    )
    # Shrink arrows and shorten all edges
    dot.edge_attr.update(arrowsize="0.5", len="0.1")
    dot.node_attr.update(shape="box", style="filled", fillcolor="white", fontname="Helvetica")

    # Encoder and first graph layer nodes (horizontal layout enforced)
    nodes = [
        ("Input",      "Input\n(12×5)",      {"shape":"ellipse"}),
        ("Enc1_relu1", "Linear+ReLU\n(5 → 128)", {}),
        ("Enc2_relu2", "Linear+ReLU\n(128 → 128)", {}),
        ("Enc3",       "Linear\n(128 → 3)",     {}),
        ("Latent",     "Latent\n(12×3)",       {"shape":"ellipse"}),
        ("GAT1",       "GATv2Conv\n(1 → 128)", {})
    ]
    for name, label, attrs in nodes:
        dot.node(name, label, **attrs)

    # Edges between encoder and first graph node with minimal length
    edges = [
        ("Input", "Enc1_relu1"),
        ("Enc1_relu1", "Enc2_relu2"),
        ("Enc2_relu2", "Enc3"),
        ("Enc3", "Latent"),
        ("Latent", "GAT1")
    ]
    for u, v in edges:
        dot.edge(u, v, len="0.1")

    # Force horizontal alignment
    with dot.subgraph() as horiz:
        horiz.attr(style="invis", rank="same")
        for name, _, _ in nodes:
            horiz.node(name)

    # Remaining graph layers cluster
    with dot.subgraph(name="cluster_graph") as graph:
        graph.attr(style="invis")
        graph.node("GAT2", "GATv2Conv\n(128 → 128)")
        graph.node("Skip", "Skip\nLinear(3 → 128)", style="dashed")
        graph.edge("GAT1", "GAT2", len="0.1")

    # Label head cluster
    with dot.subgraph(name="cluster_label") as lab:
        lab.attr(style="invis")
        lab.node("GAT3", "GATv2Conv\n(128 → 128)")
        lab.node("LabelHead", "Linear\n(128 → 4)")
        lab.node("OutputLabels", "Predicted Labels", shape="ellipse")
        for u, v in [("GAT2","GAT3"), ("GAT3","LabelHead"), ("LabelHead","OutputLabels")]:
            lab.edge(u, v, len="0.1")
        lab.edge("Skip", "GAT3", style="dotted", label="α skip", len="0.1")

    # Value head cluster
    with dot.subgraph(name="cluster_value") as val:
        val.attr(style="invis")
        val.node("GAT4", "GATv2Conv\n(128 → 128)")
        val.node("ValueHead", "Linear\n(128 → 1)")
        val.node("OutputValues", "Predicted Values", shape="ellipse")
        for u, v in [("GAT2","GAT4"), ("GAT4","ValueHead"), ("ValueHead","OutputValues")]:
            val.edge(u, v, len="0.1")
        val.edge("Skip", "GAT4", style="dotted", label="α skip", len="0.1")

    # Render to PNG file
    output_file = dot.render(output_path, view=False)
    print(f"Graph architecture saved to {output_file}")


if __name__ == "__main__":
    draw_autoencoder_architecture(dpi=300)