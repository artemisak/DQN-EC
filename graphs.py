import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import time
from typing import Dict, Tuple, Set, Optional
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
