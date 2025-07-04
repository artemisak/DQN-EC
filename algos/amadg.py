import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, distance
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import networkx as nx
from typing import Tuple, List, Set
import time

class AdaptiveManifoldDelaunayGraph:
    """
    Implementation of Adaptive Manifold-Aware Delaunay Graph (AMADG) algorithm
    for constructing graphs from point clouds optimized for Graph Attention Networks.
    """
    
    def __init__(self, k_neighbors: int = None, tau1: float = 0.3, 
                 alpha: float = 0.75, lambda_range: float = 3.0):
        """
        Initialize AMADG algorithm parameters.
        
        Args:
            k_neighbors: Number of nearest neighbors (default: min(log2(n), 20))
            tau1: Threshold for manifold compatibility score
            alpha: Preferential attachment parameter for scale-free augmentation
            lambda_range: Range parameter for long-range connections
        """
        self.k_neighbors = k_neighbors
        self.tau1 = tau1
        self.alpha = alpha
        self.lambda_range = lambda_range
        
    def construct_graph(self, points: np.ndarray) -> nx.Graph:
        """
        Construct AMADG from point cloud.
        
        Args:
            points: N x D array of point coordinates
            
        Returns:
            NetworkX graph with AMADG structure
        """
        n, d = points.shape
        
        # Set k_neighbors if not specified
        if self.k_neighbors is None:
            self.k_neighbors = min(int(np.log2(n)), 20)
            
        # Phase 1: Adaptive Local Scale Estimation
        scales = self._compute_adaptive_scales(points)
        
        # Phase 2: Weighted Delaunay Construction
        weighted_delaunay = self._construct_weighted_delaunay(points, scales)
        
        # Phase 3: Manifold-Aware Edge Filtering
        filtered_edges = self._filter_edges_manifold_aware(
            points, weighted_delaunay, scales
        )
        
        # Phase 4: Scale-Free Augmentation
        augmented_edges = self._augment_scale_free(
            points, filtered_edges, scales
        )
        
        # Construct final graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(augmented_edges)
        
        return G
    
    def _compute_adaptive_scales(self, points: np.ndarray) -> np.ndarray:
        """Phase 1: Compute adaptive scale parameter for each point."""
        n = points.shape[0]
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm='kd_tree')
        nbrs.fit(points)
        
        distances, _ = nbrs.kneighbors(points)
        # Exclude self (first neighbor) and compute mean distance
        scales = np.mean(distances[:, 1:], axis=1)
        
        return scales
    
    def _construct_weighted_delaunay(self, points: np.ndarray, 
                                   scales: np.ndarray) -> Set[Tuple[int, int]]:
        """Phase 2: Construct weighted Delaunay triangulation."""
        # For weighted Delaunay, we lift points to paraboloid
        # Standard approach: add weight as extra dimension
        n, d = points.shape
        
        # Weights are -sigma_i^2
        weights = -scales**2
        
        # Lift points: (x, y, ||x||^2 + w)
        lifted_points = np.zeros((n, d + 1))
        lifted_points[:, :d] = points
        lifted_points[:, d] = np.sum(points**2, axis=1) + weights
        
        # Compute Delaunay of lifted points
        tri = Delaunay(lifted_points)
        
        # Extract edges from simplices
        edges = set()
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)
                    
        return edges
    
    def _filter_edges_manifold_aware(self, points: np.ndarray, 
                                   edges: Set[Tuple[int, int]], 
                                   scales: np.ndarray) -> Set[Tuple[int, int]]:
        """Phase 3: Filter edges based on manifold compatibility."""
        # First, estimate local tangent spaces using PCA
        tangent_spaces = self._estimate_tangent_spaces(points)
        
        filtered_edges = set()
        
        for i, j in edges:
            # Compute manifold compatibility score
            dist_sq = np.sum((points[i] - points[j])**2)
            gaussian_term = np.exp(-dist_sq / (scales[i] * scales[j]))
            
            # Compute angle between tangent spaces
            # Using first principal component as tangent approximation
            cos_angle = abs(np.dot(tangent_spaces[i], tangent_spaces[j]))
            cos_sq = cos_angle**2
            
            M_ij = gaussian_term * cos_sq
            
            # Check if edge should be retained
            if M_ij > self.tau1 or self._is_gabriel_edge(points, i, j, edges):
                filtered_edges.add((i, j))
                
        return filtered_edges
    
    def _estimate_tangent_spaces(self, points: np.ndarray) -> np.ndarray:
        """Estimate local tangent spaces using PCA on k-neighborhoods."""
        n, d = points.shape
        tangent_spaces = np.zeros((n, d))
        
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm='kd_tree')
        nbrs.fit(points)
        
        for i in range(n):
            # Get k-nearest neighbors
            _, indices = nbrs.kneighbors([points[i]])
            neighbor_points = points[indices[0]]
            
            # Perform PCA
            pca = PCA(n_components=1)
            pca.fit(neighbor_points)
            
            # First principal component as tangent direction
            tangent_spaces[i] = pca.components_[0]
            
        return tangent_spaces
    
    def _is_gabriel_edge(self, points: np.ndarray, i: int, j: int, 
                        edges: Set[Tuple[int, int]]) -> bool:
        """Check if edge (i,j) is a Gabriel edge."""
        # Gabriel edge: no other point lies in the sphere with diameter (i,j)
        center = (points[i] + points[j]) / 2
        radius_sq = np.sum((points[i] - points[j])**2) / 4
        
        for k in range(points.shape[0]):
            if k != i and k != j:
                dist_sq = np.sum((points[k] - center)**2)
                if dist_sq < radius_sq - 1e-10:  # Numerical tolerance
                    return False
                    
        return True
    
    def _augment_scale_free(self, points: np.ndarray, 
                           edges: Set[Tuple[int, int]], 
                           scales: np.ndarray) -> Set[Tuple[int, int]]:
        """Phase 4: Add scale-free long-range connections."""
        n = points.shape[0]
        augmented_edges = edges.copy()
        
        # Compute current degrees
        degrees = np.zeros(n)
        for i, j in edges:
            degrees[i] += 1
            degrees[j] += 1
            
        # Number of edges to add per node
        m = int(np.log2(n))
        
        for i in range(n):
            # Compute sampling probabilities for all other nodes
            probs = np.zeros(n)
            
            for j in range(n):
                if i != j and (i, j) not in augmented_edges and (j, i) not in augmented_edges:
                    # Preferential attachment term
                    pref_attach = (degrees[j] + 1) ** self.alpha
                    
                    # Distance decay term
                    dist = np.linalg.norm(points[i] - points[j])
                    dist_decay = np.exp(-dist / (self.lambda_range * scales[i]))
                    
                    probs[j] = pref_attach * dist_decay
                    
            # Normalize probabilities
            if np.sum(probs) > 0:
                probs /= np.sum(probs)
                
                # Sample m edges without replacement
                num_samples = min(m, np.count_nonzero(probs))
                if num_samples > 0:
                    sampled = np.random.choice(
                        n, size=num_samples, replace=False, p=probs
                    )
                    
                    for j in sampled:
                        augmented_edges.add(tuple(sorted([i, j])))
                        degrees[i] += 1
                        degrees[j] += 1
                        
        return augmented_edges


def generate_test_data(n_points: int = 200, data_type: str = 'spiral') -> np.ndarray:
    """Generate test point cloud data."""
    if data_type == 'spiral':
        # Generate points on a 2D spiral
        t = np.linspace(0, 4 * np.pi, n_points)
        x = t * np.cos(t) / (4 * np.pi)
        y = t * np.sin(t) / (4 * np.pi)
        noise = np.random.normal(0, 0.02, (n_points, 2))
        points = np.column_stack([x, y]) + noise
        
    elif data_type == 'clusters':
        # Generate clustered data
        n_clusters = 4
        points_per_cluster = n_points // n_clusters
        points = []
        
        for i in range(n_clusters):
            center = np.random.uniform(-1, 1, 2)
            cluster_points = np.random.normal(center, 0.15, (points_per_cluster, 2))
            points.append(cluster_points)
            
        points = np.vstack(points)
        
    elif data_type == 'manifold':
        # Generate points on a 2D manifold embedded in 3D
        u = np.random.uniform(0, 2 * np.pi, n_points)
        v = np.random.uniform(0, np.pi, n_points)
        
        x = np.sin(v) * np.cos(u)
        y = np.sin(v) * np.sin(u)
        z = np.cos(v) + 0.5 * np.sin(2 * u) * np.sin(v)
        
        points = np.column_stack([x, y, z])
        
    else:
        # Random uniform distribution
        points = np.random.uniform(-1, 1, (n_points, 2))
        
    return points


def visualize_comparison(points: np.ndarray, amadg: nx.Graph, 
                        delaunay_edges: Set[Tuple[int, int]]):
    """Visualize comparison between AMADG and Delaunay triangulation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Delaunay triangulation
    ax1.scatter(points[:, 0], points[:, 1], c='blue', s=30, alpha=0.6, zorder=3)
    for i, j in delaunay_edges:
        ax1.plot([points[i, 0], points[j, 0]], 
                [points[i, 1], points[j, 1]], 
                'k-', alpha=0.3, linewidth=0.5)
    ax1.set_title('Standard Delaunay Triangulation', fontsize=14)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot AMADG
    ax2.scatter(points[:, 0], points[:, 1], c='red', s=30, alpha=0.6, zorder=3)
    
    # Color edges by type (local vs long-range)
    edge_lengths = []
    for i, j in amadg.edges():
        length = np.linalg.norm(points[i] - points[j])
        edge_lengths.append(length)
        
    median_length = np.median(edge_lengths)
    
    for (i, j), length in zip(amadg.edges(), edge_lengths):
        if length < 1.5 * median_length:
            # Local edge
            ax2.plot([points[i, 0], points[j, 0]], 
                    [points[i, 1], points[j, 1]], 
                    'b-', alpha=0.4, linewidth=0.5)
        else:
            # Long-range edge
            ax2.plot([points[i, 0], points[j, 0]], 
                    [points[i, 1], points[j, 1]], 
                    'g-', alpha=0.3, linewidth=0.5)
            
    ax2.set_title('Adaptive Manifold-Aware Delaunay Graph (AMADG)', fontsize=14)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    delaunay_graph = nx.Graph()
    delaunay_graph.add_edges_from(delaunay_edges)
    
    stats_text = (
        f"Delaunay: {len(delaunay_edges)} edges, "
        f"avg degree: {2*len(delaunay_edges)/len(points):.1f}\n"
        f"AMADG: {amadg.number_of_edges()} edges, "
        f"avg degree: {2*amadg.number_of_edges()/len(points):.1f}"
    )
    
    fig.suptitle(stats_text, fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    """Main function to demonstrate AMADG algorithm."""
    # Generate test data
    print("Generating test point cloud...")
    points = generate_test_data(n_points=150, data_type='spiral')
    
    # Construct standard Delaunay triangulation
    print("Constructing standard Delaunay triangulation...")
    start_time = time.time()
    tri = Delaunay(points)
    delaunay_edges = set()
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                edge = tuple(sorted([simplex[i], simplex[j]]))
                delaunay_edges.add(edge)
    delaunay_time = time.time() - start_time
    
    # Construct AMADG
    print("Constructing AMADG...")
    start_time = time.time()
    amadg_algo = AdaptiveManifoldDelaunayGraph()
    amadg_graph = amadg_algo.construct_graph(points)
    amadg_time = time.time() - start_time
    
    # Print timing information
    print(f"\nTiming comparison:")
    print(f"Delaunay triangulation: {delaunay_time:.4f} seconds")
    print(f"AMADG construction: {amadg_time:.4f} seconds")
    
    # Compute graph statistics
    delaunay_graph = nx.Graph()
    delaunay_graph.add_edges_from(delaunay_edges)
    
    print(f"\nGraph statistics:")
    print(f"Delaunay - Edges: {len(delaunay_edges)}, "
          f"Avg degree: {2*len(delaunay_edges)/len(points):.2f}, "
          f"Diameter: {nx.diameter(delaunay_graph) if nx.is_connected(delaunay_graph) else 'inf'}")
    print(f"AMADG - Edges: {amadg_graph.number_of_edges()}, "
          f"Avg degree: {2*amadg_graph.number_of_edges()/len(points):.2f}, "
          f"Diameter: {nx.diameter(amadg_graph) if nx.is_connected(amadg_graph) else 'inf'}")
    
    # Visualize comparison
    print("\nGenerating visualization...")
    visualize_comparison(points, amadg_graph, delaunay_edges)


if __name__ == "__main__":
    main()
