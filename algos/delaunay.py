import numpy as np
from scipy.spatial import Delaunay
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Tuple, Set, Dict, List


def construct_delaunay_graph(points: np.ndarray) -> Tuple[np.ndarray, Dict[int, Set[int]]]:
    """
    Construct a graph from a cloud of points using Delaunay triangulation.
    
    Args:
        points: Array of shape (n, 2) containing 2D points
        
    Returns:
        edges: Array of shape (m, 2) containing edge pairs
        adjacency: Dictionary mapping vertex index to set of connected vertices
    """
    # Compute Delaunay triangulation
    tri = Delaunay(points)
    
    # Extract edges efficiently using set to avoid duplicates
    edges_set = set()
    adjacency = defaultdict(set)
    
    # Iterate through simplices (triangles)
    for simplex in tri.simplices:
        # Each simplex is a triangle with 3 vertices
        # Add all 3 edges of the triangle
        for i in range(3):
            v1, v2 = simplex[i], simplex[(i + 1) % 3]
            # Store edges with smaller index first to avoid duplicates
            edge = (min(v1, v2), max(v1, v2))
            edges_set.add(edge)
            # Build adjacency list
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
    
    # Convert to numpy array for efficient storage
    edges = np.array(list(edges_set))
    
    return edges, dict(adjacency)


def visualize_delaunay_graph(points: np.ndarray, edges: np.ndarray, 
                           figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Visualize the Delaunay triangulation graph.
    
    Args:
        points: Array of shape (n, 2) containing 2D points
        edges: Array of shape (m, 2) containing edge pairs
        figsize: Figure size for matplotlib
    """
    plt.figure(figsize=figsize)
    
    # Plot edges
    for edge in edges:
        p1, p2 = points[edge[0]], points[edge[1]]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', alpha=0.6, linewidth=1)
    
    # Plot points
    plt.scatter(points[:, 0], points[:, 1], c='red', s=50, zorder=5)
    
    plt.title('Delaunay Triangulation Graph')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def generate_random_points(n: int, seed: int = 42) -> np.ndarray:
    """Generate random 2D points for testing."""
    np.random.seed(seed)
    return np.random.rand(n, 2) * 100


def compute_graph_statistics(edges: np.ndarray, adjacency: Dict[int, Set[int]]) -> Dict[str, float]:
    """Compute basic statistics about the graph."""
    num_vertices = len(adjacency)
    num_edges = len(edges)
    degrees = [len(neighbors) for neighbors in adjacency.values()]
    
    return {
        'num_vertices': num_vertices,
        'num_edges': num_edges,
        'avg_degree': np.mean(degrees),
        'min_degree': np.min(degrees),
        'max_degree': np.max(degrees)
    }


# Example usage and benchmarking
if __name__ == "__main__":
    import time
    
    # Test with different sizes
    sizes = [100, 1000, 5000]
    
    for n in sizes:
        print(f"\nTesting with {n} points:")
        
        # Generate random points
        points = generate_random_points(n)
        
        # Time the triangulation
        start_time = time.time()
        edges, adjacency = construct_delaunay_graph(points)
        elapsed_time = time.time() - start_time
        
        # Compute statistics
        stats = compute_graph_statistics(edges, adjacency)
        
        print(f"  Triangulation time: {elapsed_time:.4f} seconds")
        print(f"  Number of edges: {stats['num_edges']}")
        print(f"  Average degree: {stats['avg_degree']:.2f}")
        print(f"  Degree range: [{stats['min_degree']}, {stats['max_degree']}]")
    
    # Visualize a smaller example
    print("\nVisualizing example with 50 points...")
    points_vis = generate_random_points(50)
    edges_vis, adjacency_vis = construct_delaunay_graph(points_vis)
    visualize_delaunay_graph(points_vis, edges_vis)
    
    # Example: Using the graph for further processing
    print("\nExample graph usage:")
    print(f"Neighbors of vertex 0: {sorted(adjacency_vis[0])}")
    print(f"Total edges from vertex 0: {len(adjacency_vis[0])}")
