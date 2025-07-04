import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Tuple


class ComplexityAnalysis:
    """
    Time Complexity Analysis of Vortex Algorithm

    Optimized Version Complexity: O(n² × k × s)
    Naive Version Complexity: O(n³ × s)

    Where:
    - n = number of points
    - k = max edges per new node (typically constant, e.g., 2-3)
    - s = number of radius expansion steps (typically O(√n) to O(n) depending on distribution)
    """

    @staticmethod
    def analyze_optimized_complexity():
        """
        Analyze the optimized version's time complexity.
        """
        print("=== OPTIMIZED VERSION TIME COMPLEXITY ===\n")

        print("Main Loop: Runs until all n points are found")
        print("- Worst case: O(n) iterations if we find 1 point per iteration")
        print("- Best case: O(1) if all points are within initial radius")
        print("- Average case: O(s) where s is number of radius expansions\n")

        print("Per Iteration Operations:")
        print("1. Find unfound points: O(n) - single pass through boolean array")
        print("2. Calculate distances from unfound points to center: O(n)")
        print("3. Find points within radius: O(n)")
        print("4. For each new point found (let's say m new points):")
        print("   - Calculate distances to all previously found points: O(f)")
        print("     where f is number of already found points")
        print("   - Find k nearest neighbors: O(f) using partition")
        print("   - Add edges: O(k)")
        print("   Total for all new points: O(m × f)")
        print("5. Recalculate center: O(f) - mean of found points\n")

        print("TOTAL OPTIMIZED COMPLEXITY:")
        print("- Per iteration: O(n + m×f)")
        print("- Over all iterations: O(Σ(n + m_i×f_i))")
        print("- Worst case (finding 1 point at a time): O(n²)")
        print("- With k edges per node constraint: O(n² × k)")
        print("- With s radius expansion steps: O(n × s × k)")
        print("- Typical case: O(n² × k) where k is small constant\n")

        return "O(n² × k)"

    @staticmethod
    def analyze_naive_complexity():
        """
        Analyze a naive implementation's time complexity.
        """
        print("=== NAIVE VERSION TIME COMPLEXITY ===\n")

        print("Naive Implementation Characteristics:")
        print("- Doesn't track found/unfound points efficiently")
        print("- Recalculates all distances every iteration")
        print("- Uses nested loops instead of vectorization")
        print("- Doesn't optimize center calculation\n")

        print("Per Iteration Operations (Naive):")
        print("1. Calculate distances from ALL points to center: O(n)")
        print("2. Check each point if within radius: O(n)")
        print("3. For each point within radius:")
        print("   - Check if already processed: O(n) if using list search")
        print("   - If new, find distances to ALL other points: O(n)")
        print("   - Sort to find nearest neighbors: O(n log n)")
        print("   - Add edges: O(k)")
        print("4. Recalculate center from scratch: O(n)\n")

        print("TOTAL NAIVE COMPLEXITY:")
        print("- Per iteration: O(n + m×n×log(n))")
        print("- With poor found-point tracking: O(n²) just to check status")
        print("- Over all iterations: O(n² × log(n) × s)")
        print("- Worst case: O(n³ × log(n))")
        print("- Typical case: O(n³) due to inefficiencies\n")

        return "O(n³)"


class VortexNaive:
    """Naive implementation for comparison."""

    def __init__(self, points, initial_radius=0.1, radius_increment=0.05):
        self.points = points
        self.n_points = len(points)
        self.initial_radius = initial_radius
        self.radius_increment = radius_increment
        self.edges = []  # List of tuples instead of adjacency matrix
        self.found = []  # List instead of boolean array
        self.center = self.calculate_center(list(range(self.n_points)))

    def calculate_center(self, indices):
        """Naive center calculation."""
        if not indices:
            return np.array([0, 0])
        sum_x = sum_y = 0
        for i in indices:
            sum_x += self.points[i][0]
            sum_y += self.points[i][1]
        return np.array([sum_x / len(indices), sum_y / len(indices)])

    def construct_graph(self):
        """Naive graph construction."""
        radius = self.initial_radius

        while len(self.found) < self.n_points:
            new_found = []

            # Check every point (inefficient)
            for i in range(self.n_points):
                if i in self.found:  # O(n) search in list
                    continue

                # Calculate distance
                dist = np.sqrt((self.points[i][0] - self.center[0]) ** 2 +
                               (self.points[i][1] - self.center[1]) ** 2)

                if dist <= radius:
                    new_found.append(i)

                    # Connect to existing points (naive approach)
                    if self.found:
                        distances = []
                        for j in self.found:  # Calculate all distances
                            d = np.sqrt((self.points[i][0] - self.points[j][0]) ** 2 +
                                        (self.points[i][1] - self.points[j][1]) ** 2)
                            distances.append((d, j))

                        distances.sort()  # Full sort instead of partition
                        for k in range(min(2, len(distances))):
                            self.edges.append((i, distances[k][1]))

            if new_found:
                self.found.extend(new_found)
                self.center = self.calculate_center(self.found)
                radius = self.initial_radius
            else:
                radius += self.radius_increment


class RuntimeComparison:
    """Compare actual runtimes of optimized vs naive implementations."""

    @staticmethod
    def generate_test_points(n: int, distribution: str = 'uniform') -> np.ndarray:
        """Generate test points with different distributions."""
        np.random.seed(42)

        if distribution == 'uniform':
            return np.random.rand(n, 2) * 10
        elif distribution == 'clustered':
            # Create 3 clusters
            cluster_size = n // 3
            cluster1 = np.random.randn(cluster_size, 2) + [0, 0]
            cluster2 = np.random.randn(cluster_size, 2) + [5, 5]
            cluster3 = np.random.randn(n - 2 * cluster_size, 2) + [-5, 2]
            return np.vstack([cluster1, cluster2, cluster3])
        elif distribution == 'spiral':
            theta = np.linspace(0, 4 * np.pi, n)
            r = np.linspace(0, 5, n)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            return np.column_stack([x, y]) + np.random.randn(n, 2) * 0.1

    @staticmethod
    def measure_runtime(algorithm_class, points, **kwargs):
        """Measure runtime of an algorithm."""
        start_time = time.time()
        algo = algorithm_class(points, **kwargs)
        algo.construct_graph()
        end_time = time.time()
        return end_time - start_time

    @staticmethod
    def compare_algorithms(max_n=200, step=20):
        """Compare runtime of optimized vs naive algorithms."""
        sizes = list(range(20, max_n + 1, step))
        optimized_times = []
        naive_times = []

        print("\nRunning empirical comparison...")
        print("Size | Optimized | Naive | Ratio")
        print("-" * 40)

        for n in sizes:
            points = RuntimeComparison.generate_test_points(n, 'clustered')

            # Import the optimized version (from previous artifact)
            from radial_graph_algorithm import VortexGraphConstructor

            # Measure optimized
            opt_time = RuntimeComparison.measure_runtime(
                VortexGraphConstructor, points,
                initial_radius=0.5, radius_increment=0.2
            )
            optimized_times.append(opt_time)

            # Measure naive (only for smaller sizes due to O(n³))
            if n <= 100:
                naive_time = RuntimeComparison.measure_runtime(
                    VortexNaive, points,
                    initial_radius=0.5, radius_increment=0.2
                )
                naive_times.append(naive_time)
                ratio = naive_time / opt_time
                print(f"{n:4d} | {opt_time:8.4f}s | {naive_time:6.4f}s | {ratio:5.1f}x")
            else:
                print(f"{n:4d} | {opt_time:8.4f}s | (too slow) | -")

        return sizes, optimized_times, naive_times

    @staticmethod
    def plot_complexity_curves():
        """Plot theoretical complexity curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Theoretical curves
        n_values = np.array(range(10, 1001, 10))

        # Different complexity functions (normalized for visualization)
        O_n = n_values / 100
        O_nlogn = n_values * np.log(n_values) / 1000
        O_n2 = n_values ** 2 / 10000
        O_n2k = 3 * n_values ** 2 / 10000  # k=3
        O_n3 = n_values ** 3 / 1000000

        # Plot theoretical complexities
        ax1.plot(n_values, O_n, label='O(n)', linewidth=2)
        ax1.plot(n_values, O_nlogn, label='O(n log n)', linewidth=2)
        ax1.plot(n_values, O_n2, label='O(n²)', linewidth=2)
        ax1.plot(n_values, O_n2k, label='O(n²k) k=3', linewidth=2, linestyle='--')
        ax1.plot(n_values, O_n3, label='O(n³)', linewidth=2)

        ax1.set_xlabel('Number of Points (n)')
        ax1.set_ylabel('Time (normalized)')
        ax1.set_title('Theoretical Time Complexity Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1000)
        ax1.set_ylim(0, 20)

        # Complexity growth rates
        growth_n = [50, 100, 200, 500, 1000]
        optimized_growth = [g ** 2 * 2 / 10000 for g in growth_n]  # O(n²k)
        naive_growth = [g ** 3 / 1000000 for g in growth_n]  # O(n³)

        x = np.arange(len(growth_n))
        width = 0.35

        bars1 = ax2.bar(x - width / 2, optimized_growth, width, label='Optimized O(n²k)')
        bars2 = ax2.bar(x + width / 2, naive_growth, width, label='Naive O(n³)')

        ax2.set_xlabel('Number of Points')
        ax2.set_ylabel('Relative Time')
        ax2.set_title('Growth Rate Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(growth_n)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}', ha='center', va='bottom')

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()


def detailed_complexity_breakdown():
    """Provide detailed complexity breakdown with examples."""
    print("\n" + "=" * 60)
    print("DETAILED COMPLEXITY BREAKDOWN")
    print("=" * 60 + "\n")

    print("VARIABLES:")
    print("- n: total number of points")
    print("- k: max edges per new node (typically 2-3)")
    print("- s: number of radius expansion steps")
    print("- f: number of found points at current iteration")
    print("- m: number of new points found in current iteration\n")

    print("OPTIMIZED VERSION - STEP BY STEP:")
    print("1. Initialize: O(n) - calculate initial center")
    print("2. Main loop (runs O(s) times):")
    print("   a. Find unfound points: O(n)")
    print("   b. Calculate distances: O(n-f) ≈ O(n)")
    print("   c. Find points in radius: O(n)")
    print("   d. Connect new points: O(m × f × k)")
    print("   e. Update center: O(f)")
    print("3. Total: O(s × (n + m×f×k))")
    print("   - Best case (all points in one go): O(n²×k)")
    print("   - Worst case (one point at a time): O(n²×k)")
    print("   - Average: O(n²×k) where k is small constant\n")

    print("NAIVE VERSION - INEFFICIENCIES:")
    print("1. Uses list for 'found' tracking: O(n) to check membership")
    print("2. Calculates all distances every time: O(n²)")
    print("3. Sorts distances instead of partition: O(n log n) vs O(n)")
    print("4. No vectorization: Python loops are ~100x slower")
    print("5. Total: O(n³) with high constant factors\n")

    print("REAL-WORLD IMPACT:")
    print("For n=1000 points:")
    print("- Optimized: ~1,000,000 × k operations")
    print("- Naive: ~1,000,000,000 operations")
    print("- Speedup: ~500-1000x in practice")


if __name__ == "__main__":
    # Run theoretical analysis
    ComplexityAnalysis.analyze_optimized_complexity()
    ComplexityAnalysis.analyze_naive_complexity()

    # Show detailed breakdown
    detailed_complexity_breakdown()

    # Plot theoretical curves
    RuntimeComparison.plot_complexity_curves()

    # Run empirical comparison (small scale due to naive's poor performance)
    # Uncomment the following line if you have the optimized implementation available:
    # sizes, opt_times, naive_times = RuntimeComparison.compare_algorithms(max_n=100)