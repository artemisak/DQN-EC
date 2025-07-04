import numpy as np
from scipy.spatial import KDTree
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


class BetaSkeleton:
    def __init__(self, points: np.ndarray, beta: float = 1.0):
        """
        Initialize Beta-Skeleton graph constructor.
        
        Args:
            points: Nx2 array of 2D points
            beta: Beta parameter (default 1.0 for Gabriel graph)
        """
        self.points = np.asarray(points)
        self.n = len(self.points)
        self.beta = beta
        self.edges = []
        self.kdtree = KDTree(self.points)
        
    def _compute_lune_center_radius(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the center and radius of the beta-lune for two points.
        
        For beta >= 1: The lune is the intersection of two circles
        For beta < 1: The lune is the union of two circles
        """
        d = np.linalg.norm(p2 - p1)
        
        if self.beta >= 1:
            # For beta >= 1, we use the intersection of two circles
            # Center is the midpoint of p1 and p2
            center = (p1 + p2) / 2
            # Radius of each circle
            radius = d * self.beta / 2
        else:
            # For beta < 1, we use the union of two circles
            # This case is more complex and less common
            center = (p1 + p2) / 2
            radius = d * self.beta / 2
            
        return center, radius
    
    def _is_point_in_lune(self, point: np.ndarray, p1: np.ndarray, p2: np.ndarray, 
                          center: np.ndarray, radius: float) -> bool:
        """
        Check if a point is inside the beta-lune defined by p1 and p2.
        """
        if self.beta >= 1:
            # For beta >= 1: point is in lune if it's in both circles
            # Circle 1: centered at p1 + beta/2 * (p2-p1)
            # Circle 2: centered at p2 + beta/2 * (p1-p2)
            d12 = p2 - p1
            d_norm = np.linalg.norm(d12)
            
            c1 = p1 + (self.beta / 2) * d12
            c2 = p2 - (self.beta / 2) * d12
            r = self.beta * d_norm / 2
            
            # Check if point is in both circles
            return (np.linalg.norm(point - c1) < r and 
                    np.linalg.norm(point - c2) < r)
        else:
            # For beta < 1: point is in lune if it's in either circle
            d12 = p2 - p1
            d_norm = np.linalg.norm(d12)
            
            c1 = p1 + (self.beta / 2) * d12
            c2 = p2 - (self.beta / 2) * d12
            r = self.beta * d_norm / 2
            
            # Check if point is in either circle
            return (np.linalg.norm(point - c1) < r or 
                    np.linalg.norm(point - c2) < r)
    
    def construct_graph(self) -> List[Tuple[int, int]]:
        """
        Construct the beta-skeleton graph efficiently.
        
        Returns:
            List of edges as tuples of point indices
        """
        self.edges = []
        
        # For each pair of points
        for i in range(self.n):
            for j in range(i + 1, self.n):
                p1, p2 = self.points[i], self.points[j]
                
                # Compute lune parameters
                center, radius = self._compute_lune_center_radius(p1, p2)
                
                # Find all points that could potentially be in the lune
                # We search in a box around the lune
                d = np.linalg.norm(p2 - p1)
                search_radius = d * max(1, self.beta) / 2 + 1e-10
                
                # Get potential points using KDTree
                if self.beta >= 1:
                    # For intersection of circles, search around midpoint
                    potential_indices = self.kdtree.query_ball_point(center, search_radius)
                else:
                    # For union of circles, need to check both centers
                    d12 = p2 - p1
                    c1 = p1 + (self.beta / 2) * d12
                    c2 = p2 - (self.beta / 2) * d12
                    indices1 = self.kdtree.query_ball_point(c1, search_radius)
                    indices2 = self.kdtree.query_ball_point(c2, search_radius)
                    potential_indices = list(set(indices1 + indices2))
                
                # Check if any other point is in the lune
                is_empty = True
                for k in potential_indices:
                    if k != i and k != j:
                        if self._is_point_in_lune(self.points[k], p1, p2, center, radius):
                            is_empty = False
                            break
                
                # If lune is empty, add edge
                if is_empty:
                    self.edges.append((i, j))
        
        return self.edges
    
    def plot(self, figsize: Tuple[int, int] = (10, 10), 
             point_size: int = 50, edge_width: float = 1.0):
        """
        Plot the beta-skeleton graph.
        """
        plt.figure(figsize=figsize)
        
        # Plot edges
        for i, j in self.edges:
            plt.plot([self.points[i, 0], self.points[j, 0]], 
                    [self.points[i, 1], self.points[j, 1]], 
                    'b-', linewidth=edge_width, alpha=0.6)
        
        # Plot points
        plt.scatter(self.points[:, 0], self.points[:, 1], 
                   c='red', s=point_size, zorder=5)
        
        plt.title(f'Beta-Skeleton Graph (β = {self.beta})')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def generate_random_points(n: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate random 2D points for testing."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.rand(n, 2) * 100


# Example usage
if __name__ == "__main__":
    # Generate sample points
    points = generate_random_points(50, seed=42)
    
    # Construct beta-skeleton with beta=1 (Gabriel graph)
    beta_skeleton = BetaSkeleton(points, beta=1.0)
    edges = beta_skeleton.construct_graph()
    
    print(f"Number of points: {len(points)}")
    print(f"Number of edges: {len(edges)}")
    print(f"Beta value: {beta_skeleton.beta}")
    
    # Plot the graph
    beta_skeleton.plot()
    
    # Example with different beta values
    for beta in [0.8, 1.0, 1.5, 2.0]:
        bs = BetaSkeleton(points, beta=beta)
        edges = bs.construct_graph()
        print(f"Beta = {beta}: {len(edges)} edges")