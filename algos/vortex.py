import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Circle


class VortexGraphConstructor:
    def __init__(self, points, initial_radius=0.1, radius_increment=0.05, max_edges_per_new_node=2):
        """
        Initialize the vortex graph constructor with dynamic center recalculation.

        Parameters:
        - points: numpy array of shape (n, 2) containing 2D coordinates
        - initial_radius: starting radius for the search circle
        - radius_increment: how much to increase radius at each step
        - max_edges_per_new_node: maximum edges to create for each newly discovered node
        """
        self.points = np.array(points)
        self.n_points = len(points)
        self.initial_radius = initial_radius
        self.radius_increment = radius_increment
        self.max_edges_per_new_node = max_edges_per_new_node

        # Initialize adjacency matrix
        self.adjacency_matrix = np.zeros((self.n_points, self.n_points), dtype=int)

        # Track which points have been found
        self.found = np.zeros(self.n_points, dtype=bool)

        # Calculate initial center of mass (using ALL points)
        self.center = np.mean(self.points, axis=0)

        # Store center history for visualization
        self.center_history = [self.center.copy()]

    def construct_graph(self):
        """
        Main algorithm with dynamic center recalculation.
        """
        # Current radius
        radius = self.initial_radius

        # Keep track of all found points
        found_indices = []

        # Continue until all points are connected
        while len(found_indices) < self.n_points:
            # Calculate distances from all unfound points to current center
            unfound_mask = ~self.found
            unfound_indices = np.where(unfound_mask)[0]

            if len(unfound_indices) == 0:
                break

            unfound_points = self.points[unfound_indices]
            distances_to_center = np.linalg.norm(unfound_points - self.center, axis=1)

            # Find new points within current radius
            within_radius_mask = distances_to_center <= radius
            new_indices_local = np.where(within_radius_mask)[0]
            new_indices_global = unfound_indices[new_indices_local]

            if len(new_indices_global) > 0:
                # Connect new points to closest already-found points
                if len(found_indices) > 0:
                    for new_idx in new_indices_global:
                        new_point = self.points[new_idx]

                        # Calculate distances to all previously found points
                        found_points = self.points[found_indices]
                        distances = np.linalg.norm(found_points - new_point, axis=1)

                        # Find k nearest neighbors
                        k = min(self.max_edges_per_new_node, len(found_indices))
                        nearest_indices = np.argpartition(distances, k - 1)[:k]

                        # Add edges to adjacency matrix
                        for nearest_idx in nearest_indices:
                            found_idx = found_indices[nearest_idx]
                            self.adjacency_matrix[new_idx, found_idx] = 1
                            self.adjacency_matrix[found_idx, new_idx] = 1

                # Mark new points as found
                self.found[new_indices_global] = True
                found_indices.extend(new_indices_global)

                # RECALCULATE CENTER using only found points
                self.center = np.mean(self.points[found_indices], axis=0)
                self.center_history.append(self.center.copy())

                # Reset radius for new center
                radius = self.initial_radius
            else:
                # No new points found, increase radius
                radius += self.radius_increment

                # Safety check: if radius is too large, connect remaining points
                if radius > np.max(np.linalg.norm(self.points - self.center, axis=1)) * 2:
                    # Connect all remaining unfound points to their nearest found point
                    for idx in unfound_indices:
                        if len(found_indices) > 0:
                            distances = np.linalg.norm(
                                self.points[found_indices] - self.points[idx], axis=1
                            )
                            nearest_idx = found_indices[np.argmin(distances)]
                            self.adjacency_matrix[idx, nearest_idx] = 1
                            self.adjacency_matrix[nearest_idx, idx] = 1
                            self.found[idx] = True
                            found_indices.append(idx)
                    break

    def get_networkx_graph(self):
        """Convert adjacency matrix to NetworkX graph."""
        G = nx.Graph()

        # Add nodes with positions
        for i, point in enumerate(self.points):
            G.add_node(i, pos=(point[0], point[1]))

        # Add edges
        edges = np.argwhere(self.adjacency_matrix)
        edges = edges[edges[:, 0] < edges[:, 1]]  # Remove duplicates
        G.add_edges_from(edges)

        return G

    def visualize(self, figsize=(12, 10), show_center_path=True):
        """Visualize the constructed graph with center movement."""
        G = self.get_networkx_graph()
        pos = nx.get_node_attributes(G, 'pos')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left plot: Final graph
        nx.draw(G, pos, ax=ax1, node_color='lightblue', node_size=300,
                with_labels=True, edge_color='gray', width=1.5)

        # Draw initial center
        ax1.scatter(self.center_history[0][0], self.center_history[0][1],
                    c='green', s=200, marker='x', linewidths=3, label='Initial Center')

        # Draw final center
        ax1.scatter(self.center_history[-1][0], self.center_history[-1][1],
                    c='red', s=200, marker='x', linewidths=3, label='Final Center')

        if show_center_path and len(self.center_history) > 1:
            # Draw center movement path
            centers = np.array(self.center_history)
            ax1.plot(centers[:, 0], centers[:, 1], 'r--', alpha=0.5, linewidth=2,
                     label='Center Path')

            # Add arrows to show direction
            for i in range(len(centers) - 1):
                ax1.annotate('', xy=centers[i + 1], xytext=centers[i],
                             arrowprops=dict(arrowstyle='->', color='red', alpha=0.3))

        ax1.set_title('Vortex Graph Construction Result', fontsize=14)
        ax1.legend()
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)

        # Right plot: Order of discovery
        colors = np.zeros(self.n_points)
        for i, idx in enumerate(np.where(self.found)[0]):
            colors[idx] = i

        scatter = ax2.scatter(self.points[:, 0], self.points[:, 1],
                              c=colors, cmap='viridis', s=100)

        # Draw center path on right plot too
        if show_center_path and len(self.center_history) > 1:
            centers = np.array(self.center_history)
            ax2.plot(centers[:, 0], centers[:, 1], 'r--', alpha=0.5, linewidth=2)

        ax2.set_title('Discovery Order (darker = earlier)', fontsize=14)
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Discovery Step')

        plt.tight_layout()
        plt.show()


class AnimatedVortexGraphConstructor(VortexGraphConstructor):
    """Extended version with step-by-step animation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.animation_states = []

    def construct_graph_with_states(self):
        """Construct graph while saving states for animation."""
        radius = self.initial_radius
        found_indices = []

        # Save initial state
        self.animation_states.append({
            'radius': radius,
            'center': self.center.copy(),
            'found': self.found.copy(),
            'adjacency_matrix': self.adjacency_matrix.copy(),
            'new_indices': [],
            'found_indices': found_indices.copy()
        })

        while len(found_indices) < self.n_points:
            unfound_mask = ~self.found
            unfound_indices = np.where(unfound_mask)[0]

            if len(unfound_indices) == 0:
                break

            unfound_points = self.points[unfound_indices]
            distances_to_center = np.linalg.norm(unfound_points - self.center, axis=1)

            within_radius_mask = distances_to_center <= radius
            new_indices_local = np.where(within_radius_mask)[0]
            new_indices_global = unfound_indices[new_indices_local]

            if len(new_indices_global) > 0:
                if len(found_indices) > 0:
                    for new_idx in new_indices_global:
                        new_point = self.points[new_idx]
                        found_points = self.points[found_indices]
                        distances = np.linalg.norm(found_points - new_point, axis=1)

                        k = min(self.max_edges_per_new_node, len(found_indices))
                        nearest_indices = np.argpartition(distances, k - 1)[:k]

                        for nearest_idx in nearest_indices:
                            found_idx = found_indices[nearest_idx]
                            self.adjacency_matrix[new_idx, found_idx] = 1
                            self.adjacency_matrix[found_idx, new_idx] = 1

                self.found[new_indices_global] = True
                found_indices.extend(new_indices_global)

                # Recalculate center
                old_center = self.center.copy()
                self.center = np.mean(self.points[found_indices], axis=0)
                self.center_history.append(self.center.copy())

                # Save state
                self.animation_states.append({
                    'radius': radius,
                    'center': self.center.copy(),
                    'old_center': old_center,
                    'found': self.found.copy(),
                    'adjacency_matrix': self.adjacency_matrix.copy(),
                    'new_indices': list(new_indices_global),
                    'found_indices': found_indices.copy()
                })

                # Reset radius for new center
                radius = self.initial_radius
            else:
                radius += self.radius_increment

                if radius > np.max(np.linalg.norm(self.points - self.center, axis=1)) * 2:
                    break

    def plot_animation_frame(self, frame_idx, ax):
        """Plot a single frame of the animation."""
        ax.clear()
        state = self.animation_states[frame_idx]

        # Plot all points
        unfound_mask = ~state['found']
        ax.scatter(self.points[unfound_mask, 0], self.points[unfound_mask, 1],
                   c='lightgray', s=100, alpha=0.5, label='Unfound')

        # Plot found points (excluding new ones)
        found_mask = state['found'].copy()
        for new_idx in state['new_indices']:
            found_mask[new_idx] = False
        if np.any(found_mask):
            ax.scatter(self.points[found_mask, 0], self.points[found_mask, 1],
                       c='lightblue', s=100, label='Found')

        # Highlight new points
        if len(state['new_indices']) > 0:
            new_points = self.points[state['new_indices']]
            ax.scatter(new_points[:, 0], new_points[:, 1],
                       c='orange', s=150, label='New', zorder=5)

        # Draw edges
        edges = np.argwhere(state['adjacency_matrix'])
        for i, j in edges:
            if i < j:
                ax.plot([self.points[i, 0], self.points[j, 0]],
                        [self.points[i, 1], self.points[j, 1]],
                        'gray', linewidth=1, alpha=0.7)

        # Draw search circle
        circle = Circle(state['center'], state['radius'], fill=False,
                        edgecolor='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.add_patch(circle)

        # Draw center
        ax.scatter(state['center'][0], state['center'][1],
                   c='red', s=200, marker='x', linewidths=3, label='Current Center')

        # Draw center path
        if len(self.center_history) > 1:
            centers = np.array(self.center_history[:len(self.center_history)
            if frame_idx == len(self.animation_states) - 1
            else frame_idx + 1])
            if len(centers) > 1:
                ax.plot(centers[:, 0], centers[:, 1], 'r--', alpha=0.3, linewidth=1)

        # Set title and formatting
        n_found = np.sum(state['found'])
        ax.set_title(f'Step {frame_idx}: Center at ({state["center"][0]:.2f}, '
                     f'{state["center"][1]:.2f}), Found: {n_found}/{self.n_points}')
        ax.legend(loc='upper right')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

        # Set consistent axis limits
        margin = 1
        x_min, x_max = self.points[:, 0].min() - margin, self.points[:, 0].max() + margin
        y_min, y_max = self.points[:, 1].min() - margin, self.points[:, 1].max() + margin
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)


def demo_vortex_algorithm():
    """Demonstrate the vortex algorithm with dynamic center."""
    np.random.seed(42)

    # Create an interesting point distribution
    # Spiral pattern
    theta = np.linspace(0, 4 * np.pi, 20)
    spiral = np.column_stack([theta * np.cos(theta) * 0.15,
                              theta * np.sin(theta) * 0.15])

    # Add some clusters
    cluster1 = np.random.randn(10, 2) * 0.2 + [2, 1]
    cluster2 = np.random.randn(10, 2) * 0.2 + [-1, -1]

    points = np.vstack([spiral, cluster1, cluster2])

    # Run the algorithm
    print("Running Vortex Graph Construction Algorithm...")
    vgc = VortexGraphConstructor(
        points,
        initial_radius=0.3,
        radius_increment=0.1,
        max_edges_per_new_node=2
    )

    vgc.construct_graph()
    vgc.visualize()

    # Get statistics
    G = vgc.get_networkx_graph()
    print(f"\nGraph Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Is connected: {nx.is_connected(G)}")
    print(f"Center moved {len(vgc.center_history) - 1} times")
    print(f"Total center displacement: {np.linalg.norm(vgc.center_history[-1] - vgc.center_history[0]):.3f}")


def demo_animated_vortex():
    """Show step-by-step animation."""
    np.random.seed(123)

    # Create asymmetric distribution to show center movement
    points1 = np.random.randn(15, 2) * 0.3 + [-2, 0]
    points2 = np.random.randn(25, 2) * 0.5 + [1, 1]
    points = np.vstack([points1, points2])

    # Create animated version
    avgc = AnimatedVortexGraphConstructor(
        points,
        initial_radius=0.3,
        radius_increment=0.1,
        max_edges_per_new_node=2
    )

    avgc.construct_graph_with_states()

    # Show frames
    n_frames = len(avgc.animation_states)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Select frames to show
    frame_indices = np.linspace(0, n_frames - 1, 6, dtype=int)

    for i, frame_idx in enumerate(frame_indices):
        avgc.plot_animation_frame(frame_idx, axes[i])

    plt.suptitle('Vortex Algorithm Progress: Center Moves Toward Dense Regions', fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run basic demo
    demo_vortex_algorithm()

    # Run animated demo
    demo_animated_vortex()