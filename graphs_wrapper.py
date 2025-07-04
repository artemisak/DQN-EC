import torch
import numpy as np
from typing import Tuple, Optional, Union
from graphs import (
    GraphAlgorithm, 
    DelaunayGraphAlgorithm, 
    BetaSkeletonAlgorithm, 
    AMADGAlgorithm
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
    
    def __init__(self, beta: float = 1.0):
        self.beta = beta
        self.wrapper = TorchGraphWrapper(BetaSkeletonAlgorithm(beta=beta))
    
    def create_graph(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create Beta-skeleton graph from torch tensor points."""
        return self.wrapper.construct_graph(points)


class TorchAMADGGraph:
    """Torch-compatible AMADG graph constructor."""
    
    def __init__(self, k_neighbors: Optional[int] = None, tau1: float = 0.3,
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


# Example usage showing compatibility
if __name__ == "__main__":
    # Create sample torch tensor points
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = torch.randn(50, 3, device=device)  # 50 points in 3D
    
    # Test different graph construction methods
    print("Testing graph construction methods...")
    
    # Gabriel graph (beta=1.0)
    edge_index, edge_attr = create_gabriel_graph(points)
    print(f"Gabriel graph: {edge_index.shape[1]//2} edges")
    
    # Delaunay triangulation
    edge_index, edge_attr = create_delaunay_graph(points)
    print(f"Delaunay graph: {edge_index.shape[1]//2} edges")
    
    # Beta-skeleton with beta=1.5
    edge_index, edge_attr = create_beta_skeleton_graph(points, beta=1.5)
    print(f"Beta-skeleton (β=1.5): {edge_index.shape[1]//2} edges")
    
    # AMADG
    edge_index, edge_attr = create_amadg_graph(points)
    print(f"AMADG: {edge_index.shape[1]//2} edges")
    
    # Verify output format
    print(f"\nOutput format check:")
    print(f"edge_index shape: {edge_index.shape}, dtype: {edge_index.dtype}")
    print(f"edge_attr shape: {edge_attr.shape}, dtype: {edge_attr.dtype}")
    print(f"Device: {edge_index.device}")
