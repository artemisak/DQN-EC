import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import Delaunay
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import time
from typing import Dict, List, Tuple, Set, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
from collections import defaultdict


@dataclass
class GraphResult:
    """Unified output format for all graph algorithms."""
    edges: np.ndarray  # Shape (m, 2) array of edge pairs
    adjacency: Dict[int, Set[int]]  # Adjacency list representation
    nx_graph: nx.Graph  # NetworkX graph for advanced analysis
    construction_time: float  # Time taken to construct the graph
    algorithm_name: str  # Name of the algorithm used


class GraphAlgorithm(ABC):
    """Abstract base class for graph construction algorithms."""
    
    @abstractmethod
    def construct(self, points: np.ndarray) -> GraphResult:
        """Construct graph from points."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return algorithm name."""
        pass


class DelaunayGraphAlgorithm(GraphAlgorithm):
    """Delaunay triangulation graph algorithm."""
    
    def __init__(self):
        pass
    
    @property
    def name(self) -> str:
        return "Delaunay"
    
    def construct(self, points: np.ndarray) -> GraphResult:
        start_time = time.time()
        
        # Compute Delaunay triangulation
        tri = Delaunay(points)
        
        # Extract edges
        edges_set = set()
        adjacency = defaultdict(set)
        
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    v1, v2 = simplex[i], simplex[j]
                    edge = (min(v1, v2), max(v1, v2))
                    edges_set.add(edge)
                    adjacency[v1].add(v2)
                    adjacency[v2].add(v1)
        
        edges = np.array(list(edges_set))
        
        # Create NetworkX graph
        G = nx.Graph()
        G.add_edges_from(edges)
        
        construction_time = time.time() - start_time
        
        return GraphResult(
            edges=edges,
            adjacency=dict(adjacency),
            nx_graph=G,
            construction_time=construction_time,
            algorithm_name=self.name
        )


class BetaSkeletonAlgorithm(GraphAlgorithm):
    """Beta-skeleton graph algorithm."""
    
    def __init__(self, beta: float = 1.0):
        self.beta = beta
    
    @property
    def name(self) -> str:
        return f"Beta-Skeleton (β={self.beta})"
    
    def _is_point_in_lune(self, point: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check if a point is inside the beta-lune."""
        d12 = p2 - p1
        d_norm = np.linalg.norm(d12)
        
        if self.beta >= 1:
            # For beta >= 1: intersection of two circles
            c1 = p1 + (self.beta / 2) * d12
            c2 = p2 - (self.beta / 2) * d12
            r = self.beta * d_norm / 2
            
            return (np.linalg.norm(point - c1) < r and 
                    np.linalg.norm(point - c2) < r)
        else:
            # For beta < 1: union of two circles
            c1 = p1 + (self.beta / 2) * d12
            c2 = p2 - (self.beta / 2) * d12
            r = self.beta * d_norm / 2
            
            return (np.linalg.norm(point - c1) < r or 
                    np.linalg.norm(point - c2) < r)
    
    def construct(self, points: np.ndarray) -> GraphResult:
        start_time = time.time()
        
        n = len(points)
        kdtree = KDTree(points)
        edges = []
        adjacency = defaultdict(set)
        
        # For each pair of points
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = points[i], points[j]
                
                # Search for points that could be in the lune
                d = np.linalg.norm(p2 - p1)
                search_radius = d * max(1, self.beta) / 2 + 1e-10
                center = (p1 + p2) / 2
                
                potential_indices = kdtree.query_ball_point(center, search_radius)
                
                # Check if lune is empty
                is_empty = True
                for k in potential_indices:
                    if k != i and k != j:
                        if self._is_point_in_lune(points[k], p1, p2):
                            is_empty = False
                            break
                
                if is_empty:
                    edges.append((i, j))
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        
        edges = np.array(edges) if edges else np.array([]).reshape(0, 2)
        
        # Create NetworkX graph
        G = nx.Graph()
        if len(edges) > 0:
            G.add_edges_from(edges)
        
        construction_time = time.time() - start_time
        
        return GraphResult(
            edges=edges,
            adjacency=dict(adjacency),
            nx_graph=G,
            construction_time=construction_time,
            algorithm_name=self.name
        )


class AMADGAlgorithm(GraphAlgorithm):
    """Adaptive Manifold-Aware Delaunay Graph algorithm."""
    
    def __init__(self, k_neighbors: Optional[int] = None, tau1: float = 0.3,
                 alpha: float = 0.75, lambda_range: float = 3.0):
        self.k_neighbors = k_neighbors
        self.tau1 = tau1
        self.alpha = alpha
        self.lambda_range = lambda_range
    
    @property
    def name(self) -> str:
        return "AMADG"
    
    def construct(self, points: np.ndarray) -> GraphResult:
        start_time = time.time()
        
        n, d = points.shape
        
        # Set k_neighbors if not specified
        if self.k_neighbors is None:
            self.k_neighbors = min(int(np.log2(n)), 20)
        
        # Phase 1: Adaptive Local Scale Estimation
        scales = self._compute_adaptive_scales(points)
        
        # Phase 2: Weighted Delaunay Construction
        weighted_delaunay = self._construct_weighted_delaunay(points, scales)
        
        # Phase 3: Manifold-Aware Edge Filtering
        filtered_edges = self._filter_edges_manifold_aware(points, weighted_delaunay, scales)
        
        # Phase 4: Scale-Free Augmentation
        augmented_edges = self._augment_scale_free(points, filtered_edges, scales)
        
        # Convert to standard format
        edges = np.array(list(augmented_edges))
        adjacency = defaultdict(set)
        
        for i, j in augmented_edges:
            adjacency[i].add(j)
            adjacency[j].add(i)
        
        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(augmented_edges)
        
        construction_time = time.time() - start_time
        
        return GraphResult(
            edges=edges,
            adjacency=dict(adjacency),
            nx_graph=G,
            construction_time=construction_time,
            algorithm_name=self.name
        )
    
    # Include all the AMADG helper methods from the original code
    def _compute_adaptive_scales(self, points: np.ndarray) -> np.ndarray:
        n = points.shape[0]
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm='kd_tree')
        nbrs.fit(points)
        distances, _ = nbrs.kneighbors(points)
        scales = np.mean(distances[:, 1:], axis=1)
        return scales
    
    def _construct_weighted_delaunay(self, points: np.ndarray, 
                                   scales: np.ndarray) -> Set[Tuple[int, int]]:
        n, d = points.shape
        weights = -scales**2
        
        lifted_points = np.zeros((n, d + 1))
        lifted_points[:, :d] = points
        lifted_points[:, d] = np.sum(points**2, axis=1) + weights
        
        tri = Delaunay(lifted_points)
        
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
        tangent_spaces = self._estimate_tangent_spaces(points)
        filtered_edges = set()
        
        for i, j in edges:
            dist_sq = np.sum((points[i] - points[j])**2)
            gaussian_term = np.exp(-dist_sq / (scales[i] * scales[j]))
            cos_angle = abs(np.dot(tangent_spaces[i], tangent_spaces[j]))
            cos_sq = cos_angle**2
            M_ij = gaussian_term * cos_sq
            
            if M_ij > self.tau1 or self._is_gabriel_edge(points, i, j, edges):
                filtered_edges.add((i, j))
        
        return filtered_edges
    
    def _estimate_tangent_spaces(self, points: np.ndarray) -> np.ndarray:
        n, d = points.shape
        tangent_spaces = np.zeros((n, d))
        
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm='kd_tree')
        nbrs.fit(points)
        
        for i in range(n):
            _, indices = nbrs.kneighbors([points[i]])
            neighbor_points = points[indices[0]]
            pca = PCA(n_components=1)
            pca.fit(neighbor_points)
            tangent_spaces[i] = pca.components_[0]
        
        return tangent_spaces
    
    def _is_gabriel_edge(self, points: np.ndarray, i: int, j: int, 
                        edges: Set[Tuple[int, int]]) -> bool:
        center = (points[i] + points[j]) / 2
        radius_sq = np.sum((points[i] - points[j])**2) / 4
        
        for k in range(points.shape[0]):
            if k != i and k != j:
                dist_sq = np.sum((points[k] - center)**2)
                if dist_sq < radius_sq - 1e-10:
                    return False
        return True
    
    def _augment_scale_free(self, points: np.ndarray, 
                           edges: Set[Tuple[int, int]], 
                           scales: np.ndarray) -> Set[Tuple[int, int]]:
        n = points.shape[0]
        augmented_edges = edges.copy()
        
        degrees = np.zeros(n)
        for i, j in edges:
            degrees[i] += 1
            degrees[j] += 1
        
        m = int(np.log2(n))
        
        for i in range(n):
            probs = np.zeros(n)
            
            for j in range(n):
                if i != j and (i, j) not in augmented_edges and (j, i) not in augmented_edges:
                    pref_attach = (degrees[j] + 1) ** self.alpha
                    dist = np.linalg.norm(points[i] - points[j])
                    dist_decay = np.exp(-dist / (self.lambda_range * scales[i]))
                    probs[j] = pref_attach * dist_decay
            
            if np.sum(probs) > 0:
                probs /= np.sum(probs)
                num_samples = min(m, np.count_nonzero(probs))
                
                if num_samples > 0:
                    sampled = np.random.choice(n, size=num_samples, replace=False, p=probs)
                    
                    for j in sampled:
                        augmented_edges.add(tuple(sorted([i, j])))
                        degrees[i] += 1
                        degrees[j] += 1
        
        return augmented_edges


class UnifiedGraphFramework:
    """Unified framework for graph construction algorithms."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize framework with optional random seed."""
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = None
            
        self.algorithms: Dict[str, GraphAlgorithm] = {}
        self.results: Dict[str, GraphResult] = {}
        
    def add_algorithm(self, name: str, algorithm: GraphAlgorithm):
        """Add a graph construction algorithm."""
        self.algorithms[name] = algorithm
        
    def generate_points(self, n_points: int, data_type: str = 'uniform',
                       dimension: int = 2, **kwargs) -> np.ndarray:
        """
        Generate point cloud data.
        
        Args:
            n_points: Number of points to generate
            data_type: Type of point distribution
                - 'uniform': Uniform random distribution
                - 'gaussian': Gaussian clusters
                - 'spiral': 2D spiral pattern
                - 'manifold': 2D manifold in 3D
                - 'grid': Regular grid
                - 'circles': Concentric circles
            dimension: Dimension of points (2 or 3)
            **kwargs: Additional parameters for specific distributions
        
        Returns:
            Array of shape (n_points, dimension)
        """
        if data_type == 'uniform':
            return np.random.uniform(-1, 1, (n_points, dimension))
            
        elif data_type == 'gaussian':
            n_clusters = kwargs.get('n_clusters', 4)
            cluster_std = kwargs.get('cluster_std', 0.15)
            points = []
            
            for _ in range(n_clusters):
                center = np.random.uniform(-1, 1, dimension)
                cluster_points = np.random.normal(center, cluster_std, 
                                                (n_points // n_clusters, dimension))
                points.append(cluster_points)
            
            return np.vstack(points)
            
        elif data_type == 'spiral' and dimension == 2:
            t = np.linspace(0, 4 * np.pi, n_points)
            noise_level = kwargs.get('noise', 0.02)
            x = t * np.cos(t) / (4 * np.pi)
            y = t * np.sin(t) / (4 * np.pi)
            noise = np.random.normal(0, noise_level, (n_points, 2))
            return np.column_stack([x, y]) + noise
            
        elif data_type == 'manifold' and dimension == 3:
            u = np.random.uniform(0, 2 * np.pi, n_points)
            v = np.random.uniform(0, np.pi, n_points)
            
            x = np.sin(v) * np.cos(u)
            y = np.sin(v) * np.sin(u)
            z = np.cos(v) + 0.5 * np.sin(2 * u) * np.sin(v)
            
            return np.column_stack([x, y, z])
            
        elif data_type == 'grid':
            side_length = int(np.sqrt(n_points))
            if dimension == 2:
                x = np.linspace(-1, 1, side_length)
                y = np.linspace(-1, 1, side_length)
                xx, yy = np.meshgrid(x, y)
                return np.column_stack([xx.ravel(), yy.ravel()])[:n_points]
            else:
                side_length = int(n_points ** (1/3))
                x = np.linspace(-1, 1, side_length)
                y = np.linspace(-1, 1, side_length)
                z = np.linspace(-1, 1, side_length)
                xx, yy, zz = np.meshgrid(x, y, z)
                return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])[:n_points]
                
        elif data_type == 'circles' and dimension == 2:
            n_circles = kwargs.get('n_circles', 3)
            points = []
            
            for i in range(n_circles):
                radius = (i + 1) / n_circles
                theta = np.linspace(0, 2 * np.pi, n_points // n_circles)
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                points.append(np.column_stack([x, y]))
            
            return np.vstack(points)
            
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def run_algorithm(self, algorithm_name: str, points: np.ndarray) -> GraphResult:
        """Run a specific algorithm on points."""
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algorithm {algorithm_name} not found")
            
        result = self.algorithms[algorithm_name].construct(points)
        self.results[algorithm_name] = result
        return result
    
    def run_all_algorithms(self, points: np.ndarray) -> Dict[str, GraphResult]:
        """Run all algorithms on the same point set."""
        results = {}
        for name, algorithm in self.algorithms.items():
            print(f"Running {algorithm.name}...")
            results[name] = self.run_algorithm(name, points)
        return results
    
    def compute_metrics(self, result: GraphResult) -> Dict[str, float]:
        """Compute graph metrics."""
        G = result.nx_graph
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        metrics = {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'avg_degree': 2 * n_edges / n_nodes if n_nodes > 0 else 0,
            'density': 2 * n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0,
            'construction_time': result.construction_time,
        }
        
        # Add more metrics if graph is connected
        if nx.is_connected(G):
            metrics['diameter'] = nx.diameter(G)
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(G)
            metrics['avg_clustering'] = nx.average_clustering(G)
        else:
            components = list(nx.connected_components(G))
            metrics['n_components'] = len(components)
            metrics['largest_component_size'] = max(len(c) for c in components) if components else 0
            
        # Degree statistics
        degrees = [d for _, d in G.degree()]
        if degrees:
            metrics['min_degree'] = min(degrees)
            metrics['max_degree'] = max(degrees)
            metrics['std_degree'] = np.std(degrees)
        
        return metrics
    
    def compare_algorithms(self, points: np.ndarray) -> pd.DataFrame:
        """Compare all algorithms on the same point set."""
        results = self.run_all_algorithms(points)
        
        comparison_data = []
        for name, result in results.items():
            metrics = self.compute_metrics(result)
            metrics['algorithm'] = result.algorithm_name
            comparison_data.append(metrics)
        
        df = pd.DataFrame(comparison_data)
        df = df.set_index('algorithm')
        
        return df
    
    def plot_comparison(self, points: np.ndarray, figsize: Tuple[int, int] = (20, 10)):
        """Plot visual comparison of all algorithms."""
        n_algorithms = len(self.algorithms)
        
        if points.shape[1] == 2:
            fig, axes = plt.subplots(2, (n_algorithms + 1) // 2, figsize=figsize)
            axes = axes.flatten() if n_algorithms > 2 else [axes]
        else:
            # For 3D points, use 3D plots
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=figsize)
            axes = []
            for i in range(n_algorithms):
                ax = fig.add_subplot(2, (n_algorithms + 1) // 2, i + 1, projection='3d')
                axes.append(ax)
        
        for idx, (name, result) in enumerate(self.results.items()):
            ax = axes[idx]
            
            if points.shape[1] == 2:
                # 2D plot
                # Plot edges
                for i, j in result.edges:
                    ax.plot([points[i, 0], points[j, 0]], 
                           [points[i, 1], points[j, 1]], 
                           'b-', alpha=0.3, linewidth=0.5)
                
                # Plot points
                ax.scatter(points[:, 0], points[:, 1], c='red', s=20, zorder=5)
                ax.set_aspect('equal')
                
            else:
                # 3D plot
                # Plot edges
                for i, j in result.edges:
                    ax.plot([points[i, 0], points[j, 0]], 
                           [points[i, 1], points[j, 1]],
                           [points[i, 2], points[j, 2]], 
                           'b-', alpha=0.3, linewidth=0.5)
                
                # Plot points
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c='red', s=20, zorder=5)
            
            # Add title with key metrics
            metrics = self.compute_metrics(result)
            title = f"{result.algorithm_name}\n"
            title += f"Edges: {metrics['n_edges']}, Avg Degree: {metrics['avg_degree']:.2f}\n"
            title += f"Time: {metrics['construction_time']:.4f}s"
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Remove extra subplots if any
        for idx in range(n_algorithms, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_comparison(self, df: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)):
        """Plot bar charts comparing metrics across algorithms."""
        # Select numeric columns for plotting
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Create subplots for different metrics
        n_metrics = len(numeric_cols)
        fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            df[col].plot(kind='bar', ax=ax)
            ax.set_title(col.replace('_', ' ').title())
            ax.set_xlabel('')
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
        
        # Remove extra subplots
        for idx in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create framework instance
    framework = UnifiedGraphFramework(seed=42)
    
    # Add algorithms
    framework.add_algorithm('delaunay', DelaunayGraphAlgorithm())
    framework.add_algorithm('beta1', BetaSkeletonAlgorithm(beta=1.0))
    framework.add_algorithm('beta2', BetaSkeletonAlgorithm(beta=1.7))
    framework.add_algorithm('amadg', AMADGAlgorithm())
    
    # Generate test points
    print("Generating test points...")
    points = framework.generate_points(100, data_type='circles', dimension=2)
    
    # Compare algorithms
    print("\nComparing algorithms...")
    comparison_df = framework.compare_algorithms(points)
    print("\nMetrics Comparison:")
    print(comparison_df)
    
    # Visualize results
    print("\nPlotting visual comparison...")
    framework.plot_comparison(points)
    
    # Plot metrics comparison
    print("\nPlotting metrics comparison...")
    framework.plot_metrics_comparison(comparison_df)
