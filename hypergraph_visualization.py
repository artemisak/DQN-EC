import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull, Delaunay
from matplotlib.patches import Polygon
import random
from itertools import combinations
import colorsys

class BetaSkeletonGraph:
    """Generate beta-skeleton graphs for hypergraph construction visualization."""
    
    def __init__(self, points, beta):
        self.points = np.array(points)
        self.beta = beta
        self.n = len(points)
        self.graph = nx.Graph()
        self._construct_beta_skeleton()
    
    def _construct_beta_skeleton(self):
        """Construct beta-skeleton based on beta parameter."""
        self.graph.add_nodes_from(range(self.n))
        
        for i in range(self.n):
            self.graph.nodes[i]['pos'] = self.points[i]
        
        # For each pair of points, check beta-skeleton condition
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self._is_beta_skeleton_edge(i, j):
                    self.graph.add_edge(i, j)
    
    def _is_beta_skeleton_edge(self, i, j):
        """Check if edge (i,j) satisfies beta-skeleton condition."""
        p1, p2 = self.points[i], self.points[j]
        edge_length = np.linalg.norm(p2 - p1)
        
        if self.beta <= 1:
            # Lune-based beta-skeleton
            radius = edge_length / (2 * self.beta)
            center1 = p1 + (p2 - p1) * 0.5 + (p2 - p1)[::-1] * np.array([1, -1]) * np.sqrt(radius**2 - (edge_length/2)**2) / edge_length
            center2 = p1 + (p2 - p1) * 0.5 - (p2 - p1)[::-1] * np.array([1, -1]) * np.sqrt(radius**2 - (edge_length/2)**2) / edge_length
            
            for k in range(self.n):
                if k != i and k != j:
                    dist1 = np.linalg.norm(self.points[k] - center1)
                    dist2 = np.linalg.norm(self.points[k] - center2)
                    if min(dist1, dist2) < radius:
                        return False
        else:
            # Circle-based beta-skeleton
            radius = self.beta * edge_length / 2
            center = (p1 + p2) / 2
            
            for k in range(self.n):
                if k != i and k != j:
                    if np.linalg.norm(self.points[k] - center) < radius:
                        return False
        
        return True

class HypergraphConstructor:
    """Construct minimal hypergraph connecting planar graphs."""
    
    def __init__(self, planar_graphs):
        self.graphs = planar_graphs
        self.k = len(planar_graphs)
        self.hypergraph = []
        self.chromatic_numbers = []
        
        # Calculate chromatic numbers (approximation for planar graphs)
        for G in self.graphs:
            self.chromatic_numbers.append(self._approximate_chromatic_number(G))
        
        self.chi_max = max(self.chromatic_numbers)
        self._construct_minimal_hypergraph()
    
    def _approximate_chromatic_number(self, G):
        """Approximate chromatic number using greedy coloring."""
        if len(G.nodes()) == 0:
            return 1
        coloring = nx.greedy_color(G, strategy='largest_first')
        return max(coloring.values()) + 1
    
    def _information_exchange_capacity(self, G):
        """Calculate information exchange capacity."""
        if len(G.nodes()) == 0:
            return 1
        chi = self._approximate_chromatic_number(G)
        if len(G.nodes()) > 0:
            delta = min(dict(G.degree()).values()) if G.degree() else 0
        else:
            delta = 0
        return min(chi, delta + 1, int(np.sqrt(len(G.nodes()))))
    
    def _construct_minimal_hypergraph(self):
        """Construct minimal hypergraph according to theorem."""
        # Calculate minimal number of hyperedges needed
        min_hyperedges = max(self.chi_max, 
                           int(np.ceil(self.k * (self.k - 1) / (2 * self.chi_max))))
        
        # Generate hyperedges
        all_vertices = []
        graph_vertex_mapping = {}
        
        vertex_count = 0
        for i, G in enumerate(self.graphs):
            graph_vertex_mapping[i] = []
            for node in G.nodes():
                all_vertices.append((i, node))
                graph_vertex_mapping[i].append(vertex_count)
                vertex_count += 1
        
        # Create hyperedges connecting different graphs
        self.hypergraph = []
        connected_pairs = set()
        
        for _ in range(min_hyperedges):
            hyperedge = set()
            
            # Ensure each hyperedge connects at least 2 different graphs
            selected_graphs = random.sample(range(self.k), min(self.k, random.randint(2, 4)))
            
            for graph_idx in selected_graphs:
                if graph_vertex_mapping[graph_idx]:
                    # Select vertices respecting chromatic constraints
                    available_vertices = graph_vertex_mapping[graph_idx][:self.chromatic_numbers[graph_idx]]
                    if available_vertices:
                        vertex = random.choice(available_vertices)
                        hyperedge.add(vertex)
            
            if len(hyperedge) >= 2:
                self.hypergraph.append(hyperedge)
                
                # Track connected pairs
                graph_indices = list(set(all_vertices[v][0] for v in hyperedge))
                for i, j in combinations(graph_indices, 2):
                    connected_pairs.add((min(i,j), max(i,j)))
        
        # Ensure all pairs are connected
        all_pairs = set((i, j) for i in range(self.k) for j in range(i+1, self.k))
        missing_pairs = all_pairs - connected_pairs
        
        for i, j in missing_pairs:
            hyperedge = set()
            if graph_vertex_mapping[i]:
                hyperedge.add(random.choice(graph_vertex_mapping[i]))
            if graph_vertex_mapping[j]:
                hyperedge.add(random.choice(graph_vertex_mapping[j]))
            if len(hyperedge) >= 2:
                self.hypergraph.append(hyperedge)

def generate_random_points(n, region_bounds):
    """Generate random points within specified bounds."""
    x_min, x_max, y_min, y_max = region_bounds
    points = []
    for _ in range(n):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        points.append([x, y])
    return points

def visualize_hypergraph_construction():
    """Main visualization function."""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate three beta-skeleton graphs with random beta values
    regions = [
        (0, 3, 0, 3),    # Region 1
        (5, 8, 0, 3),    # Region 2  
        (2, 5, 5, 8),    # Region 3
    ]
    
    graphs = []
    beta_values = []
    
    # Create three beta-skeleton graphs
    for i, region in enumerate(regions):
        n_points = random.randint(8, 15)
        points = generate_random_points(n_points, region)
        beta = random.uniform(1.0, 2.0)
        beta_values.append(beta)
        
        beta_graph = BetaSkeletonGraph(points, beta)
        graphs.append(beta_graph.graph)
    
    # Construct hypergraph
    hypergraph_constructor = HypergraphConstructor(graphs)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    # fig.suptitle('Minimal Hypergraph Construction for Beta-Skeleton Graphs', fontsize=16, fontweight='bold')
    
    # Colors for different graphs
    colors = ['red', 'blue', 'green']
    
    # Plot individual beta-skeleton graphs
    axes = [ax1, ax2, ax3]
    for i, (G, ax) in enumerate(zip(graphs, axes)):
        pos = nx.get_node_attributes(G, 'pos')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors[i], 
                              node_size=100, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                              width=1, alpha=0.6)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        
        ax.set_title(f'Beta-Skeleton Graph {i+1}\n(β={beta_values[i]:.2f}, χ≈{hypergraph_constructor.chromatic_numbers[i]})', 
                    fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Plot combined hypergraph
    ax4.set_title('Combined Hypergraph with Minimal Connections', fontweight='bold')
    
    # Draw all original graphs
    vertex_positions = {}
    vertex_colors = {}
    vertex_count = 0
    
    for i, G in enumerate(graphs):
        pos = nx.get_node_attributes(G, 'pos')
        for node in G.nodes():
            vertex_positions[vertex_count] = pos[node]
            vertex_colors[vertex_count] = colors[i]
            vertex_count += 1
    
    # Draw vertices
    for vertex, pos in vertex_positions.items():
        ax4.scatter(pos[0], pos[1], c=vertex_colors[vertex], s=100, alpha=0.8, edgecolors='black')
        ax4.annotate(str(vertex), pos, xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Draw original edges
    vertex_count = 0
    for i, G in enumerate(graphs):
        pos = nx.get_node_attributes(G, 'pos')
        for edge in G.edges():
            start_pos = pos[edge[0]]
            end_pos = pos[edge[1]]
            ax4.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                    color='gray', alpha=0.3, linewidth=1)
        vertex_count += len(G.nodes())
    
    # Draw hyperedges
    hyperedge_colors = plt.cm.Set3(np.linspace(0, 1, len(hypergraph_constructor.hypergraph)))
    
    for i, hyperedge in enumerate(hypergraph_constructor.hypergraph):
        if len(hyperedge) >= 2:
            # Draw hyperedge as a convex hull
            hyperedge_positions = [vertex_positions[v] for v in hyperedge]
            if len(hyperedge_positions) >= 3:
                # try:
                #     hull = ConvexHull(hyperedge_positions)
                #     hull_points = [hyperedge_positions[vertex] for vertex in hull.vertices]
                #     polygon = Polygon(hull_points, alpha=0.2, facecolor=hyperedge_colors[i], 
                #                     edgecolor=hyperedge_colors[i], linewidth=2)
                #     ax4.add_patch(polygon)
                # except:
                # If convex hull fails, draw as star
                center = np.mean(hyperedge_positions, axis=0)
                for pos in hyperedge_positions:
                    ax4.plot([center[0], pos[0]], [center[1], pos[1]], 
                            color=hyperedge_colors[i], linewidth=2, alpha=0.7)
            else:
                # For 2 vertices, draw a thick line
                pos1, pos2 = hyperedge_positions[0], hyperedge_positions[1]
                ax4.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                        color=hyperedge_colors[i], linewidth=3, alpha=0.8)
    
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    # Add theoretical information
    theoretical_min = max(hypergraph_constructor.chi_max, 
                         int(np.ceil(hypergraph_constructor.k * (hypergraph_constructor.k - 1) / 
                                   (2 * hypergraph_constructor.chi_max))))
    
    info_text = f"""Theoretical Analysis:
• Max chromatic number: {hypergraph_constructor.chi_max}
• Number of graphs: {hypergraph_constructor.k}
• Theoretical minimum hyperedges: {theoretical_min}
• Actual hyperedges constructed: {len(hypergraph_constructor.hypergraph)}
• Chromatic numbers: {hypergraph_constructor.chromatic_numbers}"""
    
    fig.text(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print("=== HYPERGRAPH CONSTRUCTION ANALYSIS ===")
    print(f"Beta values used: {[f'{b:.3f}' for b in beta_values]}")
    print(f"Graph sizes: {[len(G.nodes()) for G in graphs]}")
    print(f"Chromatic numbers: {hypergraph_constructor.chromatic_numbers}")
    print(f"Maximum chromatic number: {hypergraph_constructor.chi_max}")
    print(f"Theoretical minimum hyperedges needed: {theoretical_min}")
    print(f"Actual hyperedges constructed: {len(hypergraph_constructor.hypergraph)}")
    print(f"Hyperedge details:")
    for i, he in enumerate(hypergraph_constructor.hypergraph):
        print(f"  Hyperedge {i+1}: {he} (size: {len(he)})")
    
    return hypergraph_constructor, beta_values

# Run the visualization
if __name__ == "__main__":
    hypergraph_constructor, beta_values = visualize_hypergraph_construction()