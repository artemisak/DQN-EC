import torch
from typing import Tuple, Optional
from graphs import (
    GraphAlgorithm, 
    DelaunayGraphAlgorithm, 
    BetaSkeletonAlgorithm, 
    AMADGAlgorithm,
    AnisotropicDelaunayAlgorithm,
    DALGGAlgorithm
)


class TorchGraphWrapper:
    """Wrapper to make graph algorithms from graphs.py compatible with PyTorch tensors."""
    
    def __init__(self, algorithm: GraphAlgorithm):
        """
        Initialize wrapper with a graph algorithm.
        
        Args:
            algorithm: Instance of GraphAlgorithm (e.g., DelaunayGraphAlgorithm)
        """
        self.algorithm = algorithm
    
    def construct_graph(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct graph from torch tensor points, matching create_gabriel_graph interface.
        
        Args:
            points: Torch tensor of shape (num_points, dimensions)
            
        Returns:
            edge_index: Torch tensor of shape (2, num_edges) with dtype torch.long
            edge_attr: Torch tensor of shape (num_edges,) containing edge distances
        """
        # Store device and convert to numpy
        device = points.device
        points_np = points.detach().cpu().numpy()
        
        # Construct graph using the algorithm
        result = self.algorithm.construct(points_np)
        
        # Extract edges
        edges = result.edges
        
        # Build edge list in the format expected by PyTorch Geometric
        edge_indices = []
        edge_attrs = []
        
        if len(edges) > 0:
            for i, j in edges:
                # Add both directions for undirected graph
                edge_indices.append([i, j])
                edge_indices.append([j, i])
                
                # Calculate distance as edge attribute
                dist = torch.norm(points[i] - points[j])
                edge_attrs.append(dist)
                edge_attrs.append(dist)
        
        # Convert to tensors
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long, device=device).t()
            edge_attr = torch.stack(edge_attrs)
        else:
            # Handle empty graph case
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0,), dtype=torch.float, device=device)
        
        return edge_index, edge_attr


class TorchDelaunayGraph:
    """Torch-compatible Delaunay graph constructor."""
    
    def __init__(self):
        self.wrapper = TorchGraphWrapper(DelaunayGraphAlgorithm())
    
    def create_graph(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create Delaunay graph from torch tensor points."""
        return self.wrapper.construct_graph(points)


class TorchBetaSkeletonGraph:
    """Torch-compatible Beta-skeleton graph constructor."""
    
    def __init__(self, beta: float = 1.7):
        self.wrapper = TorchGraphWrapper(BetaSkeletonAlgorithm(beta=beta))
    
    def create_graph(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create Beta-skeleton graph from torch tensor points."""
        return self.wrapper.construct_graph(points)


class TorchAnisotropicDelaunayGraph:
    """Torch-compatible Anisotropic Delaunay graph constructor."""

    def __init__(self, pca_neighbors: int = 3, length_weight: float = 1.2, alignment_weight: float = 0.8):
        self.wrapper = TorchGraphWrapper(AnisotropicDelaunayAlgorithm(pca_neighbors=pca_neighbors,
                                                                      length_weight=length_weight,
                                                                      alignment_weight=alignment_weight))

    def create_graph(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create Anisotropic Delaunay graph from torch tensor points."""
        return self.wrapper.construct_graph(points)


class TorchDALGG:
    """Torch-compatible DALGG graph constructor."""

    def __init__(self, k_density: int = 3, alpha: float = 1.2, beta: float = 0.8):
        self.wrapper = TorchGraphWrapper(DALGGAlgorithm(k_density=k_density, alpha=alpha, beta=beta))

    def create_graph(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create DALGG graph from torch tensor points."""
        return self.wrapper.construct_graph(points)


class TorchAMADGGraph:
    """Torch-compatible AMADG graph constructor."""
    
    def __init__(self, k_neighbors: int = 3, tau1: float = 0.3,
                 alpha: float = 0.75, lambda_range: float = 3.0):
        self.wrapper = TorchGraphWrapper(
            AMADGAlgorithm(k_neighbors=k_neighbors, tau1=tau1, 
                          alpha=alpha, lambda_range=lambda_range)
        )
    
    def create_graph(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create AMADG graph from torch tensor points."""
        return self.wrapper.construct_graph(points)


# Drop-in replacement for create_gabriel_graph
def create_gabriel_graph(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create Gabriel graph (Beta-skeleton with beta=1.0) from torch tensor points.
    This is a drop-in replacement for the create_gabriel_graph method.
    
    Args:
        points: Torch tensor of shape (num_points, dimensions)
        
    Returns:
        edge_index: Torch tensor of shape (2, num_edges) with dtype torch.long
        edge_attr: Torch tensor of shape (num_edges,) containing edge distances
    """
    gabriel_graph = TorchBetaSkeletonGraph(beta=1.0)
    return gabriel_graph.create_graph(points)


# Alternative graph constructors matching the same interface
def create_delaunay_graph(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create Delaunay triangulation graph from torch tensor points."""
    delaunay_graph = TorchDelaunayGraph()
    return delaunay_graph.create_graph(points)


def create_beta_skeleton_graph(points: torch.Tensor, beta: float = 1.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create Beta-skeleton graph from torch tensor points."""
    beta_graph = TorchBetaSkeletonGraph(beta=beta)
    return beta_graph.create_graph(points)


def create_amadg_graph(points: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create AMADG graph from torch tensor points."""
    amadg_graph = TorchAMADGGraph(**kwargs)
    return amadg_graph.create_graph(points)


def create_anisotropic_graph(points: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create Anisotropic graph from torch tensor points."""
    anisotropic_graph = TorchAMADGGraph(**kwargs)
    return anisotropic_graph.create_graph(points)


def create_dal_graph(points: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create DALGG graph from torch tensor points."""
    dal_graph = TorchDALGG(**kwargs)
    return dal_graph.create_graph(points)
