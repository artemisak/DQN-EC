import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import time
from typing import Dict, Tuple, Set, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
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

    def __init__(self, beta: float = 1.7):
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


class AnisotropicDelaunayAlgorithm(GraphAlgorithm):
    def __init__(self, pca_neighbors: int = 3, length_weight: float = 1.2, alignment_weight: float = 0.8):
        self.k = pca_neighbors
        self.alpha = length_weight
        self.beta = alignment_weight

    @property
    def name(self) -> str:
        return "AnisotropicDelaunayRefinement"

    def construct(self, points: np.ndarray) -> GraphResult:
        start_time = time.time()
        tri = Delaunay(points)
        edge_set = set()

        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                edge = tuple(sorted((a, b)))
                edge_set.add(edge)

        tree = KDTree(points)
        edge_scores = []
        edges = list(edge_set)

        for a, b in edges:
            p1, p2 = points[a], points[b]
            vec = p2 - p1
            dist = np.linalg.norm(vec)

            # Use KD-tree for efficient neighbor lookup
            dists, neighbors_idx = tree.query(p1, k=self.k + 1)
            neighbors_idx = neighbors_idx[1:]  # exclude point itself
            neighborhood = points[neighbors_idx]

            if neighborhood.shape[0] >= 2:
                pca = PCA(n_components=1)
                pca.fit(neighborhood)
                direction = pca.components_[0]
                unit_vec = vec / (np.linalg.norm(vec) + 1e-8)
                alignment = np.abs(np.dot(unit_vec, direction))
            else:
                alignment = 0.5

            score = self.alpha * dist + self.beta * (1 - alignment)
            edge_scores.append((score, (a, b)))

        edge_scores.sort()
        keep_count = int(len(edge_scores) * 0.8)
        refined_edges = [e for _, e in edge_scores[:keep_count]]

        adjacency = defaultdict(set)
        for u, v in refined_edges:
            adjacency[u].add(v)
            adjacency[v].add(u)

        nx_g = nx.Graph()
        nx_g.add_edges_from(refined_edges)

        return GraphResult(
            edges=np.array(refined_edges),
            adjacency=dict(adjacency),
            nx_graph=nx_g,
            construction_time=time.time() - start_time,
            algorithm_name=self.name,
        )


class KNNGPAlgorithm(GraphAlgorithm):
    """
    Density-Adaptive Local Geometric Graph algorithm.

    This algorithm constructs a sparse graph with well-connected cliques by:
    1. Estimating local density using k-nearest neighbors
    2. Adapting neighborhood size based on local density
    3. Pruning edges using a modified Gabriel criterion

    References:
        - Gabriel, K.R. and Sokal, R.R. (1969). "A new statistical approach to geographic variation analysis"
        - Ester et al. (1996). "A density-based algorithm for discovering clusters"
        - Von Luxburg (2007). "A tutorial on spectral clustering"
    """
    def __init__(self, k_density: int = 6, alpha: float = 1.8, beta: float = 0.88):
        """
        Initialize KNNGP algorithm.

        Args:
            1. k_density: Number of neighbors for density estimation
                - Higher k_density → More stable density estimates
                - Lower k_density → More sensitive to local variations

            2. alpha: Density scaling factor for adaptive neighborhood
                - alpha = 0 → No adaptation (uniform k for all nodes)
                - alpha = 1 → Linear scaling with density
                - alpha > 1 → Exponential scaling (more extreme adaptation)

            3. beta: Geometric pruning threshold (0 < beta <= 1)
                - beta < 1 → Stricter pruning (fewer edges)
                - beta = 1 → Standard Gabriel graph
                - beta > 1 → Relaxed pruning (more edges)
        """
        self.k_density = k_density
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-10  # Small constant to avoid division by zero

    @property
    def name(self) -> str:
        return f"KNNGP(k={self.k_density},α={self.alpha},β={self.beta})"

    def construct(self, points: np.ndarray) -> GraphResult:
        """
        Construct graph from points using KNNGP algorithm.

        Args:
            points: n × d array of point coordinates

        Returns:
            GraphResult containing the constructed graph
        """
        start_time = time.time()

        n_points = len(points)

        # Build KD-tree for efficient nearest neighbor queries
        kdtree = KDTree(points)

        # Phase 1: Local Density Estimation
        densities = self._estimate_local_densities(points, kdtree)

        # Phase 2: Adaptive Neighborhood Construction with Gabriel Pruning
        edges_set = set()
        adjacency = defaultdict(set)

        for i in range(n_points):
            # Get adaptive neighbors for point i
            neighbors = self._get_adaptive_neighbors(i, points, kdtree, densities)

            # Phase 3: Apply Gabriel criterion to prune edges
            for j in neighbors:
                if i < j:  # Avoid duplicate edges
                    if self._should_connect_gabriel(i, j, points, kdtree):
                        edges_set.add((i, j))
                        adjacency[i].add(j)
                        adjacency[j].add(i)

        # Convert to numpy array
        edges = np.array(list(edges_set))

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

    def _estimate_local_densities(self, points: np.ndarray, kdtree: KDTree) -> np.ndarray:
        """
        Estimate local density for each point using k-nearest neighbors.

        Args:
            points: Array of point coordinates
            kdtree: KD-tree for efficient nearest neighbor queries

        Returns:
            Array of density values for each point
        """
        n_points = len(points)
        densities = np.zeros(n_points)

        for i in range(n_points):
            # Query k+1 neighbors (including the point itself)
            distances, _ = kdtree.query(points[i], k=min(self.k_density + 1, n_points))

            # Local density as inverse of average k-NN distance
            # Exclude the first distance (0, distance to itself)
            avg_distance = np.mean(distances[1:]) if len(distances) > 1 else self.epsilon
            densities[i] = 1.0 / (avg_distance + self.epsilon)

        return densities

    def _get_adaptive_neighbors(self,
                                point_idx: int,
                                points: np.ndarray,
                                kdtree: KDTree,
                                densities: np.ndarray) -> List[int]:
        """
        Get adaptive number of neighbors based on local density.

        Args:
            point_idx: Index of the point
            points: Array of all points
            kdtree: KD-tree for neighbor queries
            densities: Array of density values

        Returns:
            List of neighbor indices
        """
        median_density = np.median(densities)

        # Adaptive k based on local density
        k_adaptive = int(self.k_density * (densities[point_idx] / median_density) ** self.alpha)

        # Bound the adaptive k to reasonable values
        k_adaptive = min(max(k_adaptive, 3), 2 * self.k_density)
        k_adaptive = min(k_adaptive, len(points) - 1)  # Can't have more neighbors than points

        # Query neighbors
        _, neighbors = kdtree.query(points[point_idx], k=k_adaptive + 1)

        # Return neighbors excluding the point itself
        return neighbors[1:].tolist()

    def _should_connect_gabriel(self,
                                i: int,
                                j: int,
                                points: np.ndarray,
                                kdtree: KDTree) -> bool:
        """
        Modified Gabriel criterion with density awareness.

        An edge (i,j) is included if no other point k lies in the β-influence region,
        which is defined as the lens-shaped region where both d(i,k) < β*d(i,j)
        and d(j,k) < β*d(i,j).

        Args:
            i, j: Indices of the two points
            points: Array of all points
            kdtree: KD-tree for efficient spatial queries

        Returns:
            True if edge should be included, False otherwise
        """
        p_i, p_j = points[i], points[j]
        d_ij = np.linalg.norm(p_i - p_j)

        # Find points that could potentially be in the influence region
        # We only need to check points within β*d_ij distance from both i and j
        search_radius = self.beta * d_ij

        # Get potential interfering points near i
        indices_near_i = kdtree.query_ball_point(p_i, search_radius)

        # Check if any point violates the Gabriel criterion
        for k in indices_near_i:
            if k == i or k == j:
                continue

            p_k = points[k]
            d_ik = np.linalg.norm(p_i - p_k)
            d_jk = np.linalg.norm(p_j - p_k)

            # β-Gabriel condition: point k should not be in the lens
            if d_ik < self.beta * d_ij and d_jk < self.beta * d_ij:
                return False

        return True


class NormalizedKNNGPAlgorithm(GraphAlgorithm):
    """
    Improved Density-Adaptive Local Geometric Graph algorithm.

    Changes from original:
    1. Better density estimation using normalized densities
    2. More sophisticated adaptive neighborhood calculation
    3. Improved Gabriel criterion with distance weighting
    4. Added edge weight calculation for better graph quality
    """

    def __init__(self, k_density: int = 6, alpha: float = 0.5, beta: float = 1.0,
                 min_neighbors: int = 3, max_neighbor_ratio: float = 3.0):
        """
        Initialize Normalized KNNGP algorithm.

        Args:
            k_density: Number of neighbors for density estimation (increased default)
            alpha: Density scaling factor for adaptive neighborhood (increased for more adaptation)
            beta: Geometric pruning threshold (0 < beta <= 1) (increased for less pruning)
            min_neighbors: Minimum number of neighbors to maintain connectivity
            max_neighbor_ratio: Maximum ratio of adaptive neighbors to k_density
        """
        self.k_density = k_density
        self.alpha = alpha
        self.beta = beta
        self.min_neighbors = min_neighbors
        self.max_neighbor_ratio = max_neighbor_ratio
        self.epsilon = 1e-10

    @property
    def name(self) -> str:
        return f"NormalizedKNNGP(k={self.k_density},α={self.alpha},β={self.beta})"

    def construct(self, points: np.ndarray) -> GraphResult:
        start_time = time.time()

        n_points = len(points)

        # Build KD-tree for efficient nearest neighbor queries
        kdtree = KDTree(points)

        # Phase 1: Improved Local Density Estimation
        densities = self._estimate_local_densities_improved(points, kdtree)

        # Phase 2: Adaptive Neighborhood Construction
        edges_set = set()
        adjacency = defaultdict(set)
        edge_weights = {}

        # Track degree for each node to ensure minimum connectivity
        degrees = np.zeros(n_points)

        for i in range(n_points):
            # Get adaptive neighbors for point i
            neighbors = self._get_adaptive_neighbors_improved(i, points, kdtree, densities)

            # Sort neighbors by distance for prioritization
            neighbor_dists = [(j, np.linalg.norm(points[i] - points[j])) for j in neighbors]
            neighbor_dists.sort(key=lambda x: x[1])

            # Phase 3: Apply improved Gabriel criterion
            added_count = 0
            for j, dist in neighbor_dists:
                if i < j:  # Avoid duplicate edges
                    if self._should_connect_gabriel_improved(i, j, points, kdtree, densities):
                        edge = (i, j)
                        edges_set.add(edge)
                        adjacency[i].add(j)
                        adjacency[j].add(i)

                        # Calculate edge weight based on density and distance
                        weight = self._calculate_edge_weight(i, j, points, densities)
                        edge_weights[edge] = weight

                        degrees[i] += 1
                        degrees[j] += 1
                        added_count += 1

                        # Early stopping if we have enough connections
                        if added_count >= 2 * self.k_density:
                            break

        # Phase 4: Ensure minimum connectivity
        for i in range(n_points):
            if degrees[i] < self.min_neighbors:
                # Find closest neighbors not yet connected
                dists, indices = kdtree.query(points[i], k=min(n_points, 2 * self.k_density))

                for idx in range(1, len(indices)):  # Skip self
                    j = indices[idx]
                    if j not in adjacency[i] and degrees[i] < self.min_neighbors:
                        edge = tuple(sorted([i, j]))
                        if edge not in edges_set:
                            edges_set.add(edge)
                            adjacency[i].add(j)
                            adjacency[j].add(i)

                            weight = self._calculate_edge_weight(i, j, points, densities)
                            edge_weights[edge] = weight

                            degrees[i] += 1
                            degrees[j] += 1

        # Convert to numpy array
        edges = np.array(list(edges_set)) if edges_set else np.array([]).reshape(0, 2)

        # Create NetworkX graph
        G = nx.Graph()
        if len(edges) > 0:
            G.add_edges_from(edges)
            nx.set_edge_attributes(G, edge_weights, 'weight')

        construction_time = time.time() - start_time

        return GraphResult(
            edges=edges,
            adjacency=dict(adjacency),
            nx_graph=G,
            construction_time=construction_time,
            algorithm_name=self.name
        )

    def _estimate_local_densities_improved(self, points: np.ndarray, kdtree: KDTree) -> np.ndarray:
        """
        Improved density estimation with normalization and outlier handling.
        """
        n_points = len(points)
        densities = np.zeros(n_points)

        # Use adaptive k based on dataset size
        k_adaptive = min(self.k_density + int(np.log2(n_points)), n_points - 1)

        for i in range(n_points):
            # Query k+1 neighbors (including the point itself)
            distances, _ = kdtree.query(points[i], k=min(k_adaptive + 1, n_points))

            if len(distances) > 1:
                # Use harmonic mean for more robust density estimation
                # This is less sensitive to outliers than arithmetic mean
                avg_distance = len(distances[1:]) / np.sum(1.0 / (distances[1:] + self.epsilon))
                densities[i] = 1.0 / (avg_distance + self.epsilon)
            else:
                densities[i] = 1.0 / self.epsilon

        # Normalize densities using log transform to handle outliers
        densities = np.log1p(densities)
        densities = (densities - np.min(densities)) / (np.max(densities) - np.min(densities) + self.epsilon)

        return densities

    def _get_adaptive_neighbors_improved(self,
                                         point_idx: int,
                                         points: np.ndarray,
                                         kdtree: KDTree,
                                         densities: np.ndarray) -> List[int]:
        """
        Improved adaptive neighborhood with better bounds and density consideration.
        """
        # Use percentile-based density Metrics for robustness
        density_percentile = np.percentile(densities, 50)
        density_ratio = (densities[point_idx] + self.epsilon) / (density_percentile + self.epsilon)

        # Adaptive k with smoother scaling
        k_adaptive = int(self.k_density * (density_ratio ** self.alpha))

        # Improved bounds
        k_adaptive = max(k_adaptive, self.min_neighbors)
        k_adaptive = min(k_adaptive, int(self.k_density * self.max_neighbor_ratio))
        k_adaptive = min(k_adaptive, len(points) - 1)

        # Query neighbors
        _, neighbors = kdtree.query(points[point_idx], k=k_adaptive + 1)

        return neighbors[1:].tolist()

    def _should_connect_gabriel_improved(self,
                                         i: int,
                                         j: int,
                                         points: np.ndarray,
                                         kdtree: KDTree,
                                         densities: np.ndarray) -> bool:
        """
        Improved Gabriel criterion with density-aware modifications.
        """
        p_i, p_j = points[i], points[j]
        d_ij = np.linalg.norm(p_i - p_j)

        # Density-adjusted beta: be more permissive in dense regions
        avg_density = (densities[i] + densities[j]) / 2
        adjusted_beta = self.beta + (1 - self.beta) * (1 - avg_density) * 0.5

        # Search radius
        search_radius = adjusted_beta * d_ij

        # Check midpoint region more efficiently
        midpoint = (p_i + p_j) / 2
        candidates = kdtree.query_ball_point(midpoint, search_radius)

        for k in candidates:
            if k == i or k == j:
                continue

            p_k = points[k]
            d_ik = np.linalg.norm(p_i - p_k)
            d_jk = np.linalg.norm(p_j - p_k)

            # Modified Gabriel condition with density weighting
            # Points in denser regions are less likely to block edges
            density_factor = 1.0 - 0.3 * densities[k]  # Higher density points block less

            if d_ik < adjusted_beta * d_ij * density_factor and \
                    d_jk < adjusted_beta * d_ij * density_factor:
                return False

        return True

    def _calculate_edge_weight(self, i: int, j: int, points: np.ndarray,
                               densities: np.ndarray) -> float:
        """
        Calculate edge weight based on distance and density.
        """
        dist = np.linalg.norm(points[i] - points[j])
        avg_density = (densities[i] + densities[j]) / 2

        # Weight favors short edges in dense regions
        weight = dist / (1 + avg_density)

        return weight


class KNNGraphAlgorithm(GraphAlgorithm):
    """
    k-Nearest Neighbors (kNN) graph algorithm.

    This algorithm connects each point to its k nearest neighbors based on
    Euclidean distance. The resulting graph can be either directed or undirected.

    In the directed version, an edge from i to j exists if j is among the k
    nearest neighbors of i. In the undirected version, an edge exists if either
    i is a k-NN of j OR j is a k-NN of i (mutual kNN).

    References:
    - Cover, T. and Hart, P. (1967). "Nearest neighbor pattern classification"
    - Bentley, J.L. (1975). "Multidimensional binary search trees used for
      associative searching"
    """

    def __init__(self, k_neighbors: int = 3, metric: str = 'euclidean',
                 include_self: bool = False, mutual: bool = True):
        """
        Initialize kNN graph algorithm.

        Args:
            k_neighbors: Number of nearest neighbors to connect
            metric: Distance metric to use (Euclidean by default)
            include_self: Whether to include self-loops
            mutual: If True, creates undirected graph where edge exists if either
                   node is in the other's k-NN. If False, directed graph.
        """
        self.k = k_neighbors
        self.metric = metric
        self.include_self = include_self
        self.mutual = mutual

    @property
    def name(self) -> str:
        mutual_str = "mutual" if self.mutual else "directed"
        return f"kNN(k={self.k},{mutual_str})"

    def construct(self, points: np.ndarray) -> GraphResult:
        """
        Construct kNN graph from points.

        Args:
            points: n × d array of point coordinates

        Returns:
            GraphResult containing the constructed graph
        """
        start_time = time.time()

        n_points = len(points)

        # Adjust k if needed
        k_actual = min(self.k, n_points - 1) if not self.include_self else min(self.k, n_points)

        # Compute k nearest neighbors
        nbrs = NearestNeighbors(
            n_neighbors=k_actual + (0 if self.include_self else 1),
            algorithm='auto',
            metric=self.metric
        )
        nbrs.fit(points)

        distances, indices = nbrs.kneighbors(points)

        # Build edge set and adjacency list
        edges_set = set()
        adjacency = defaultdict(set)

        for i in range(n_points):
            # Get neighbors for point i
            neighbor_indices = indices[i]
            neighbor_distances = distances[i]

            # Skip self if not including self-loops
            start_idx = 0 if self.include_self else 1

            for idx in range(start_idx, len(neighbor_indices)):
                j = neighbor_indices[idx]

                if self.mutual:
                    # Undirected graph: add edge as tuple (min, max)
                    edge = tuple(sorted([i, j]))
                    edges_set.add(edge)
                    adjacency[i].add(j)
                    adjacency[j].add(i)

                else:
                    # Directed graph: add edge as (i, j)
                    edge = (i, j)
                    edges_set.add(edge)
                    adjacency[i].add(j)

        # Convert edges to numpy array
        edges = np.array(list(edges_set))

        # Create NetworkX graph
        if self.mutual:
            G = nx.Graph()
        else:
            G = nx.DiGraph()

        G.add_nodes_from(range(n_points))

        G.add_edges_from(edges_set)

        construction_time = time.time() - start_time

        return GraphResult(
            edges=edges,
            adjacency=dict(adjacency),
            nx_graph=G,
            construction_time=construction_time,
            algorithm_name=self.name
        )


class FullyConnectedGraphAlgorithm(GraphAlgorithm):
    """
    Fully Connected (Complete) graph algorithm.

    This algorithm creates a complete graph where every node is connected
    to every other node. For n nodes, this results in n(n-1)/2 edges.

    Complete graphs are useful for:
    - Testing graph algorithms on dense graphs
    - Baseline comparisons
    - Theoretical analysis
    - Small-scale problems where all pairwise relationships matter

    Warning: The number of edges grows quadratically with the number of nodes,
    so this should only be used for relatively small point sets.
    """

    def __init__(self, include_self: bool = False):
        """
        Initialize Fully Connected graph algorithm.

        Args:
            include_self: Whether to include edges from nodes to themselves
        """
        self.include_self_loops = include_self

    @property
    def name(self) -> str:
        suffix = " (with self-loops)" if self.include_self_loops else ""
        return f"FullyConnected{suffix}"

    def construct(self, points: np.ndarray) -> GraphResult:
        """
        Construct a fully connected graph from points.

        Args:
            points: n × d array of point coordinates

        Returns:
            GraphResult containing the constructed complete graph
        """
        start_time = time.time()

        n_points = len(points)
        edges_list = []
        adjacency = defaultdict(set)

        # Generate all possible edges
        for i in range(n_points):
            # Start from i+1 to avoid duplicate edges in undirected graph
            # Start from i if self-loops are included
            start_j = i if self.include_self_loops else i + 1

            for j in range(start_j, n_points):
                if i != j:  # Regular edge
                    edges_list.append((i, j))
                    adjacency[i].add(j)
                    adjacency[j].add(i)
                elif self.include_self_loops:  # Self-loop
                    edges_list.append((i, i))
                    adjacency[i].add(i)

        # Convert to numpy array
        edges = np.array(edges_list) if edges_list else np.array([]).reshape(0, 2)

        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(n_points))
        G.add_edges_from(edges_list)

        # Add node positions as attributes for visualization
        pos_dict = {i: points[i] for i in range(n_points)}
        nx.set_node_attributes(G, pos_dict, 'pos')

        construction_time = time.time() - start_time

        # Log some statistics
        expected_edges = n_points * (n_points - 1) // 2
        if self.include_self_loops:
            expected_edges += n_points

        assert len(edges) == expected_edges, \
            f"Expected {expected_edges} edges but got {len(edges)}"

        return GraphResult(
            edges=edges,
            adjacency=dict(adjacency),
            nx_graph=G,
            construction_time=construction_time,
            algorithm_name=self.name
        )
