import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


def visualize_algorithm_steps():
    """Create a visual breakdown of algorithm steps and their complexities."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Optimized Algorithm Flow
    ax1.text(0.5, 0.95, 'OPTIMIZED VORTEX ALGORITHM',
             ha='center', va='top', fontsize=16, weight='bold', transform=ax1.transAxes)

    # Define steps and their complexities
    opt_steps = [
        ("Initialize center from all points", "O(n)", 0.1),
        ("Main Loop (s iterations)", "", 0.2),
        ("  Find unfound points", "O(n)", 0.05),
        ("  Calculate distances to center", "O(n-f)", 0.05),
        ("  Find points within radius", "O(n)", 0.05),
        ("  For each new point:", "", 0.02),
        ("    Calculate distances to found", "O(f)", 0.03),
        ("    Find k nearest (partition)", "O(f)", 0.03),
        ("    Add edges", "O(k)", 0.01),
        ("  Recalculate center", "O(f)", 0.02),
        ("  Reset/increase radius", "O(1)", 0.01),
        ("TOTAL", "O(n² × k)", 0.15)
    ]

    y_pos = 0.85
    for step, complexity, height in opt_steps:
        # Draw rectangle for time proportion
        if complexity and complexity != "":
            rect = Rectangle((0.7, y_pos - height / 2), 0.25, height,
                             facecolor='lightblue', edgecolor='black')
            ax1.add_patch(rect)

        # Add text
        indent = "  " if step.startswith("  ") else ""
        ax1.text(0.05, y_pos, step, va='center', fontsize=10)
        if complexity:
            ax1.text(0.65, y_pos, complexity, va='center', ha='right',
                     fontsize=10, weight='bold' if step == "TOTAL" else 'normal')

        y_pos -= height + 0.02

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # Naive Algorithm Flow
    ax2.text(0.5, 0.95, 'NAIVE VORTEX ALGORITHM',
             ha='center', va='top', fontsize=16, weight='bold', transform=ax2.transAxes)

    naive_steps = [
        ("Initialize center from all points", "O(n)", 0.1),
        ("Main Loop (many iterations)", "", 0.2),
        ("  For each point (all n):", "", 0.05),
        ("    Check if in found list", "O(n)", 0.08),
        ("    Calculate distance to center", "O(1)", 0.02),
        ("    If within radius:", "", 0.02),
        ("      For each found point:", "", 0.02),
        ("        Calculate distance", "O(1)", 0.01),
        ("      Sort all distances", "O(n log n)", 0.12),
        ("      Add edges", "O(k)", 0.01),
        ("  Recalculate center (loop)", "O(n)", 0.05),
        ("  Increase radius", "O(1)", 0.01),
        ("TOTAL", "O(n³)", 0.15)
    ]

    y_pos = 0.85
    for step, complexity, height in naive_steps:
        # Draw rectangle for time proportion
        if complexity and complexity != "":
            color = 'lightcoral' if 'O(n' in complexity else 'lightyellow'
            rect = Rectangle((0.7, y_pos - height / 2), 0.25, height,
                             facecolor=color, edgecolor='black')
            ax2.add_patch(rect)

        # Add text
        ax2.text(0.05, y_pos, step, va='center', fontsize=10)
        if complexity:
            ax2.text(0.65, y_pos, complexity, va='center', ha='right',
                     fontsize=10, weight='bold' if step == "TOTAL" else 'normal')

        y_pos -= height + 0.02

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    plt.tight_layout()
    return fig


def plot_complexity_comparison():
    """Create comprehensive complexity comparison plots."""
    fig = plt.figure(figsize=(15, 10))

    # Create grid
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])

    # Plot 1: Growth comparison
    ax1 = fig.add_subplot(gs[0, :])
    n_values = np.logspace(1, 3.5, 50)  # 10 to ~3000

    # Calculate complexities (with realistic constants)
    optimized = 0.001 * n_values ** 2 * 2  # k=2
    naive = 0.0001 * n_values ** 3

    ax1.loglog(n_values, optimized, 'b-', linewidth=3, label='Optimized: O(n²k)')
    ax1.loglog(n_values, naive, 'r-', linewidth=3, label='Naive: O(n³)')

    # Add shaded regions
    ax1.fill_between(n_values, optimized * 0.5, optimized * 2, alpha=0.2, color='blue')
    ax1.fill_between(n_values, naive * 0.5, naive * 2, alpha=0.2, color='red')

    ax1.set_xlabel('Number of Points (n)', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Time Complexity Growth Comparison (Log-Log Scale)', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # Add annotations
    ax1.annotate('Quadratic growth\n(manageable)', xy=(500, 0.5), xytext=(200, 2),
                 arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))
    ax1.annotate('Cubic growth\n(becomes prohibitive)', xy=(500, 12.5), xytext=(800, 5),
                 arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    # Plot 2: Breakdown by operation
    ax2 = fig.add_subplot(gs[1, 0])
    operations = ['Distance\nCalc', 'Finding\nNeighbors', 'Center\nUpdate', 'Edge\nCreation']
    optimized_ops = [30, 40, 10, 20]  # Percentage of time
    naive_ops = [20, 50, 15, 15]

    x = np.arange(len(operations))
    width = 0.35

    bars1 = ax2.bar(x - width / 2, optimized_ops, width, label='Optimized', color='lightblue')
    bars2 = ax2.bar(x + width / 2, naive_ops, width, label='Naive', color='lightcoral')

    ax2.set_ylabel('% of Total Time', fontsize=12)
    ax2.set_title('Time Distribution by Operation', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(operations)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Scalability limits
    ax3 = fig.add_subplot(gs[1, 1])
    sizes = [100, 500, 1000, 5000, 10000]
    opt_times = [0.01, 0.25, 1, 25, 100]  # seconds
    naive_times = [0.1, 12.5, 100, 12500, 100000]  # seconds

    # Convert large times to hours
    naive_times_display = []
    labels = []
    for s, t in zip(sizes, naive_times):
        if t < 60:
            naive_times_display.append(t)
            labels.append(f'{s}')
        elif t < 3600:
            naive_times_display.append(t / 60)
            labels.append(f'{s}\n({t / 60:.1f}m)')
        else:
            naive_times_display.append(t / 3600)
            labels.append(f'{s}\n({t / 3600:.1f}h)')

    x = np.arange(len(sizes))
    ax3.semilogy(x, opt_times, 'bo-', linewidth=2, markersize=8, label='Optimized')
    ax3.semilogy(x[:3], naive_times[:3], 'ro-', linewidth=2, markersize=8, label='Naive')
    ax3.semilogy(x[2:], naive_times[2:], 'ro--', linewidth=1, markersize=8, alpha=0.5)

    ax3.set_xlabel('Number of Points', fontsize=12)
    ax3.set_ylabel('Time (seconds)', fontsize=12)
    ax3.set_title('Practical Runtime Limits', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add feasibility line
    ax3.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='1 minute')
    ax3.axhline(y=3600, color='red', linestyle='--', alpha=0.7, label='1 hour')

    # Plot 4: Memory complexity
    ax4 = fig.add_subplot(gs[2, :])
    n_vals = np.array([100, 1000, 10000, 100000])

    # Memory usage in MB
    adjacency_matrix = n_vals ** 2 * 8 / 1e6  # 8 bytes per float
    edge_list = n_vals * 4 * 8 / 1e6  # Assume average degree of 4
    auxiliary = n_vals * 100 / 1e6  # Other arrays and structures

    width = 0.6
    x = np.arange(len(n_vals))

    p1 = ax4.bar(x, adjacency_matrix, width, label='Adjacency Matrix', color='lightblue')
    p2 = ax4.bar(x, edge_list, width, bottom=adjacency_matrix, label='Edge Storage', color='lightgreen')
    p3 = ax4.bar(x, auxiliary, width, bottom=adjacency_matrix + edge_list, label='Auxiliary', color='lightyellow')

    ax4.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax4.set_xlabel('Number of Points', fontsize=12)
    ax4.set_title('Memory Complexity Comparison', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{n:,}' for n in n_vals])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Add memory annotations
    for i, (adj, total) in enumerate(zip(adjacency_matrix, adjacency_matrix + edge_list + auxiliary)):
        if total > 1000:
            ax4.text(i, total + 100, f'{total / 1000:.1f} GB', ha='center', fontsize=10)
        else:
            ax4.text(i, total + 50, f'{total:.0f} MB', ha='center', fontsize=10)

    plt.tight_layout()
    return fig


def create_complexity_summary_table():
    """Create a summary table of complexities."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Table data
    headers = ['Aspect', 'Optimized Algorithm', 'Naive Algorithm', 'Improvement Factor']
    data = [
        ['Time Complexity', 'O(n² × k)', 'O(n³)', 'n/k ≈ n/3'],
        ['Space Complexity', 'O(n²)', 'O(n²)', '1x'],
        ['Distance Calculations', 'O(n) per iteration', 'O(n²) per iteration', 'n'],
        ['Neighbor Finding', 'O(n) partition', 'O(n log n) sort', 'log n'],
        ['Center Update', 'O(f) incremental', 'O(n) full recalc', 'n/f'],
        ['Practical Limit', '~10,000 points', '~500 points', '20x'],
        ['1000 points time', '~1 second', '~100 seconds', '100x'],
        ['Implementation', 'Vectorized numpy', 'Python loops', '~10x from vectorization']
    ]

    # Create table
    table = ax.table(cellText=data, colLabels=headers, loc='center',
                     cellLoc='center', colWidths=[0.2, 0.25, 0.25, 0.2])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color code
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(data) + 1):
        table[(i, 0)].set_facecolor('#E8F5E9')
        table[(i, 0)].set_text_props(weight='bold')

        # Highlight improvement factors
        if i <= len(data):
            table[(i, 3)].set_facecolor('#FFEB3B')

    ax.set_title('Vortex Algorithm Complexity Comparison Summary',
                 fontsize=16, weight='bold', pad=20)

    return fig


# Execute all visualizations
if __name__ == "__main__":
    # Create algorithm flow visualization
    fig1 = visualize_algorithm_steps()
    plt.show()

    # Create complexity comparison plots
    fig2 = plot_complexity_comparison()
    plt.show()

    # Create summary table
    fig3 = create_complexity_summary_table()
    plt.show()