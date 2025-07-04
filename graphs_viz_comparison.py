import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Tuple, Optional
from graphs import (GraphAlgorithm, GraphResult,
                    DelaunayGraphAlgorithm, BetaSkeletonAlgorithm,
                    AMADGAlgorithm)


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
                side_length = int(n_points ** (1 / 3))
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