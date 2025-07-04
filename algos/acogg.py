import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay, KDTree
from scipy.sparse.linalg import eigsh
from sklearn.random_projection import GaussianRandomProjection
import networkx as nx
from typing import Tuple, List, Dict, Set
import heapq
from dataclasses import dataclass
import warnings


@dataclass
class ACOGGConfig:
    """Configuration parameters for ACOGG algorithm"""
    jl_embedding_dim_factor: float = 5.0  # O(log N) multiplier for JL embedding
    commute_time_threshold_percentile: float = 90.0  # Percentile for bottleneck identification
    small_world_prob_decay: float = 2.0  # Distance decay for small-world connections
    max_augmentation_degree: int = 5  # Max additional edges per node
    spectral_refinement_iterations: int = 3
    spectral_gap_target: float = 0.1  # Target spectral gap
    min_edge_weight: float = 1e-6  # Minimum edge weight to avoid numerical issues


class ACOGG:
    """Adaptive Commute-Optimized Gabriel Graph construction algorithm"""

    def __init__(self, config: ACOGGConfig = None):
        self.config = config or ACOGGConfig()
        self.points = None
        self.n_points = 0
        self.kdtree = None
        self.graph = None
        self.edge_weights = {}

    def fit_transform(self, points: np.ndarray) -> sp.csr_matrix:
        """
        Construct ACOGG from point cloud

        Args:
            points: 2D array of shape (n_points, 2)

        Returns:
            Sparse adjacency matrix of the constructed graph
        """
        self.points = np.asarray(points, dtype=np.float64)
        self.n_points = len(self.points)

        if self.n_points < 3:
            raise ValueError("Need at least 3 points")

        # Phase 1: Spatial indexing and Gabriel graph construction
        print("Phase 1: Constructing Gabriel graph...")
        gabriel_edges = self._construct_gabriel_graph()

        # Initialize graph with Gabriel edges
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.n_points))
        for i, j in gabriel_edges:
            dist = np.linalg.norm(self.points[i] - self.points[j])
            self.graph.add_edge(i, j, weight=1.0 / max(dist, self.config.min_edge_weight))

        # Phase 2: Commute time analysis and bottleneck identification
        print("Phase 2: Analyzing commute times...")
        bottleneck_pairs = self._identify_bottlenecks()

        # Phase 3: Hierarchical small-world augmentation
        print("Phase 3: Adding small-world connections...")
        self._augment_small_world(bottleneck_pairs)

        # Phase 4: Spectral feedback optimization
        print("Phase 4: Spectral optimization...")
        self._spectral_refinement()

        # Convert to sparse matrix
        return self._graph_to_sparse_matrix()

    def _construct_gabriel_graph(self) -> List[Tuple[int, int]]:
        """Phase 1: Construct Gabriel graph using Delaunay triangulation as superset"""
        # Build KD-tree for efficient spatial queries
        self.kdtree = KDTree(self.points)

        # Get Delaunay triangulation as superset of Gabriel graph
        tri = Delaunay(self.points)

        gabriel_edges = []
        checked_edges = set()

        # Check each Delaunay edge for Gabriel property
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    if edge in checked_edges:
                        continue
                    checked_edges.add(edge)

                    if self._is_gabriel_edge(edge[0], edge[1]):
                        gabriel_edges.append(edge)

        return gabriel_edges

    def _is_gabriel_edge(self, i: int, j: int) -> bool:
        """Check if edge (i,j) satisfies Gabriel graph property"""
        # Center and radius of circle with diameter ij
        center = (self.points[i] + self.points[j]) / 2
        radius = np.linalg.norm(self.points[i] - self.points[j]) / 2

        # Find all points within the circle
        candidates_idx = self.kdtree.query_ball_point(center, radius + 1e-10)

        # Check if any point (other than i,j) lies inside the circle
        for k in candidates_idx:
            if k == i or k == j:
                continue
            if np.linalg.norm(self.points[k] - center) < radius - 1e-10:
                return False

        return True

    def _identify_bottlenecks(self) -> List[Tuple[int, int]]:
        """Phase 2: Identify bottlenecks using approximate commute times"""
        # Get approximate effective resistances using JL embedding
        effective_resistances = self._approximate_effective_resistances()

        # Find node pairs with high commute times
        threshold = np.percentile(list(effective_resistances.values()),
                                  self.config.commute_time_threshold_percentile)

        bottleneck_pairs = []
        for (i, j), resistance in effective_resistances.items():
            if resistance > threshold and not self.graph.has_edge(i, j):
                bottleneck_pairs.append((i, j, resistance))

        # Sort by resistance (descending)
        bottleneck_pairs.sort(key=lambda x: x[2], reverse=True)

        return [(i, j) for i, j, _ in bottleneck_pairs[:len(bottleneck_pairs) // 2]]

    def _approximate_effective_resistances(self) -> Dict[Tuple[int, int], float]:
        """Approximate effective resistances using Johnson-Lindenstrauss embedding"""
        # Convert graph to Laplacian matrix
        L = nx.laplacian_matrix(self.graph, weight='weight').astype(np.float64)

        # Compute pseudoinverse approximation via JL embedding
        embedding_dim = int(self.config.jl_embedding_dim_factor * np.log(self.n_points))
        embedding_dim = min(embedding_dim, self.n_points - 1)

        # Use random projection for dimensionality reduction
        transformer = GaussianRandomProjection(n_components=embedding_dim, random_state=42)

        # Sample node pairs for resistance computation
        n_samples = min(self.n_points * int(np.log(self.n_points)), self.n_points ** 2 // 4)
        sample_pairs = []

        # Include all disconnected components
        components = list(nx.connected_components(self.graph))
        if len(components) > 1:
            for i, comp1 in enumerate(components):
                for j in range(i + 1, len(components)):
                    comp2 = components[j]
                    # Sample pairs between components
                    for _ in range(min(5, len(comp1) * len(comp2))):
                        node1 = np.random.choice(list(comp1))
                        node2 = np.random.choice(list(comp2))
                        sample_pairs.append((node1, node2))

        # Random sampling for remaining pairs
        while len(sample_pairs) < n_samples:
            i, j = np.random.randint(0, self.n_points, 2)
            if i != j:
                sample_pairs.append(tuple(sorted([i, j])))

        sample_pairs = list(set(sample_pairs))[:n_samples]

        # Approximate resistances
        resistances = {}

        # For connected components, use actual computation
        if nx.is_connected(self.graph):
            try:
                # Use sparse eigendecomposition for efficiency
                eigenvalues, eigenvectors = eigsh(L.astype(np.float64),
                                                  k=min(embedding_dim, self.n_points - 2),
                                                  which='SM', sigma=1e-10)

                # Filter out near-zero eigenvalues
                mask = eigenvalues > 1e-10
                eigenvalues = eigenvalues[mask]
                eigenvectors = eigenvectors[:, mask]

                for i, j in sample_pairs:
                    diff = eigenvectors[i] - eigenvectors[j]
                    resistance = np.sum(diff ** 2 / eigenvalues)
                    resistances[(i, j)] = resistance
            except:
                # Fallback to graph distance
                for i, j in sample_pairs:
                    try:
                        dist = nx.shortest_path_length(self.graph, i, j)
                        resistances[(i, j)] = dist
                    except:
                        resistances[(i, j)] = float('inf')
        else:
            # For disconnected graphs, use large resistance for disconnected pairs
            for i, j in sample_pairs:
                try:
                    dist = nx.shortest_path_length(self.graph, i, j)
                    resistances[(i, j)] = dist
                except:
                    resistances[(i, j)] = 1000.0  # Large but finite resistance

        return resistances

    def _augment_small_world(self, bottleneck_pairs: List[Tuple[int, int]]):
        """Phase 3: Add small-world connections using hierarchical approach"""
        # Track augmentation degree per node
        augmentation_degree = {i: 0 for i in range(self.n_points)}

        # Build spatial hierarchy levels
        hierarchy_levels = self._build_spatial_hierarchy()

        # Add edges for bottleneck pairs first
        for i, j in bottleneck_pairs:
            if augmentation_degree[i] < self.config.max_augmentation_degree and \
                    augmentation_degree[j] < self.config.max_augmentation_degree:
                dist = np.linalg.norm(self.points[i] - self.points[j])
                weight = 1.0 / max(dist, self.config.min_edge_weight)
                self.graph.add_edge(i, j, weight=weight)
                augmentation_degree[i] += 1
                augmentation_degree[j] += 1

        # Add hierarchical small-world connections
        for level, nodes in enumerate(hierarchy_levels[1:], 1):  # Skip first level
            # Sample connections at this level
            n_connections = max(1, len(nodes) // (2 ** level))

            for _ in range(n_connections):
                if len(nodes) < 2:
                    break

                # Sample two nodes from this level
                i, j = np.random.choice(nodes, 2, replace=False)

                if not self.graph.has_edge(i, j) and \
                        augmentation_degree[i] < self.config.max_augmentation_degree and \
                        augmentation_degree[j] < self.config.max_augmentation_degree:

                    dist = np.linalg.norm(self.points[i] - self.points[j])
                    # Probability decreases with distance
                    prob = 1.0 / (dist ** self.config.small_world_prob_decay)

                    if np.random.random() < prob:
                        weight = 1.0 / max(dist, self.config.min_edge_weight)
                        self.graph.add_edge(i, j, weight=weight)
                        augmentation_degree[i] += 1
                        augmentation_degree[j] += 1

    def _build_spatial_hierarchy(self) -> List[List[int]]:
        """Build spatial hierarchy using recursive spatial partitioning"""
        # Simple grid-based hierarchy
        n_levels = int(np.log2(self.n_points)) + 1
        hierarchy = []

        # Level 0: all nodes
        hierarchy.append(list(range(self.n_points)))

        # Build subsequent levels
        for level in range(1, min(n_levels, 8)):  # Cap at 8 levels
            grid_size = 2 ** level
            nodes_at_level = []

            # Discretize space into grid
            x_min, y_min = self.points.min(axis=0)
            x_max, y_max = self.points.max(axis=0)

            x_bins = np.linspace(x_min, x_max, grid_size + 1)
            y_bins = np.linspace(y_min, y_max, grid_size + 1)

            # Select representative nodes from each cell
            for i in range(grid_size):
                for j in range(grid_size):
                    # Find nodes in this cell
                    mask = (self.points[:, 0] >= x_bins[i]) & \
                           (self.points[:, 0] < x_bins[i + 1]) & \
                           (self.points[:, 1] >= y_bins[j]) & \
                           (self.points[:, 1] < y_bins[j + 1])

                    cell_nodes = np.where(mask)[0]
                    if len(cell_nodes) > 0:
                        # Select node closest to cell center as representative
                        cell_center = [(x_bins[i] + x_bins[i + 1]) / 2,
                                       (y_bins[j] + y_bins[j + 1]) / 2]
                        distances = np.linalg.norm(self.points[cell_nodes] - cell_center, axis=1)
                        representative = cell_nodes[np.argmin(distances)]
                        nodes_at_level.append(representative)

            hierarchy.append(nodes_at_level)

        return hierarchy

    def _spectral_refinement(self):
        """Phase 4: Refine graph using spectral feedback"""
        for iteration in range(self.config.spectral_refinement_iterations):
            # Compute spectral gap
            L = nx.laplacian_matrix(self.graph, weight='weight').astype(np.float64)

            try:
                # Get first few eigenvalues
                eigenvalues = eigsh(L, k=min(3, self.n_points - 1),
                                    which='SM', return_eigenvectors=False)
                eigenvalues = sorted(eigenvalues)

                if len(eigenvalues) >= 2:
                    spectral_gap = eigenvalues[1] - eigenvalues[0]
                else:
                    spectral_gap = 0.0

            except:
                # Skip spectral refinement if computation fails
                warnings.warn("Spectral computation failed, skipping refinement")
                break

            # If spectral gap is satisfactory, stop
            if spectral_gap >= self.config.spectral_gap_target:
                break

            # Identify poorly connected regions
            if len(eigenvalues) >= 2 and eigenvalues[1] > 1e-10:
                try:
                    _, eigenvectors = eigsh(L, k=2, which='SM')
                    fiedler = eigenvectors[:, 1]

                    # Nodes with extreme Fiedler values are in different clusters
                    sorted_nodes = np.argsort(fiedler)
                    bottom_nodes = sorted_nodes[:len(sorted_nodes) // 4]
                    top_nodes = sorted_nodes[-len(sorted_nodes) // 4:]

                    # Add connections between clusters
                    n_additions = min(5, len(bottom_nodes), len(top_nodes))
                    for _ in range(n_additions):
                        i = np.random.choice(bottom_nodes)
                        j = np.random.choice(top_nodes)

                        if not self.graph.has_edge(i, j):
                            dist = np.linalg.norm(self.points[i] - self.points[j])
                            weight = 1.0 / max(dist, self.config.min_edge_weight)
                            self.graph.add_edge(i, j, weight=weight)
                except:
                    break

    def _graph_to_sparse_matrix(self) -> sp.csr_matrix:
        """Convert NetworkX graph to sparse adjacency matrix"""
        # Get edges and weights
        edges = list(self.graph.edges())
        weights = [self.graph[u][v]['weight'] for u, v in edges]

        # Build sparse matrix
        row_ind = [e[0] for e in edges] + [e[1] for e in edges]
        col_ind = [e[1] for e in edges] + [e[0] for e in edges]
        data = weights + weights  # Symmetric matrix

        adj_matrix = sp.csr_matrix((data, (row_ind, col_ind)),
                                   shape=(self.n_points, self.n_points))

        return adj_matrix

    def get_graph_stats(self) -> Dict:
        """Get statistics about the constructed graph"""
        stats = {
            'n_nodes': self.n_points,
            'n_edges': self.graph.number_of_edges(),
            'avg_degree': 2 * self.graph.number_of_edges() / self.n_points,
            'n_components': nx.number_connected_components(self.graph),
            'density': nx.density(self.graph)
        }

        # Compute diameter for largest component
        if nx.is_connected(self.graph):
            stats['diameter'] = nx.diameter(self.graph)
        else:
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            stats['diameter'] = nx.diameter(subgraph)
            stats['largest_component_size'] = len(largest_cc)

        return stats


# Example usage and visualization
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate sample 2D point cloud
    np.random.seed(42)

    # Create clusters with different densities
    cluster1 = np.random.randn(30, 2) * 0.5 + [-2, -2]
    cluster2 = np.random.randn(30, 2) * 0.5 + [2, 2]
    cluster3 = np.random.randn(20, 2) * 0.3 + [0, 3]
    sparse_points = np.random.uniform(-4, 4, (20, 2))

    points = np.vstack([cluster1, cluster2, cluster3, sparse_points])

    # Construct ACOGG
    print("Constructing ACOGG...")
    acogg = ACOGG()
    adj_matrix = acogg.fit_transform(points)

    # Print statistics
    stats = acogg.get_graph_stats()
    print("\nGraph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Visualize the graph
    plt.figure(figsize=(12, 5))

    # Plot original Gabriel graph
    plt.subplot(1, 2, 1)
    gabriel_edges = acogg._construct_gabriel_graph()
    for i, j in gabriel_edges:
        plt.plot([points[i, 0], points[j, 0]],
                 [points[i, 1], points[j, 1]], 'b-', alpha=0.3, linewidth=0.5)
    plt.scatter(points[:, 0], points[:, 1], c='red', s=20, zorder=5)
    plt.title("Gabriel Graph (Initial)")
    plt.axis('equal')

    # Plot final ACOGG
    plt.subplot(1, 2, 2)
    edges = list(acogg.graph.edges())
    for i, j in edges:
        # Color edges by whether they're Gabriel edges or augmented
        if (i, j) in gabriel_edges or (j, i) in gabriel_edges:
            plt.plot([points[i, 0], points[j, 0]],
                     [points[i, 1], points[j, 1]], 'b-', alpha=0.3, linewidth=0.5)
        else:
            plt.plot([points[i, 0], points[j, 0]],
                     [points[i, 1], points[j, 1]], 'g--', alpha=0.5, linewidth=1)
    plt.scatter(points[:, 0], points[:, 1], c='red', s=20, zorder=5)
    plt.title("ACOGG (Final)")
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

    # Compare with k-NN graph
    from sklearn.neighbors import kneighbors_graph

    k = int(stats['avg_degree'])
    knn_graph = kneighbors_graph(points, n_neighbors=k, mode='distance')

    print(f"\nComparison with {k}-NN graph:")
    print(f"  ACOGG edges: {stats['n_edges']}")
    print(f"  k-NN edges: {knn_graph.nnz // 2}")  # Divide by 2 for undirected