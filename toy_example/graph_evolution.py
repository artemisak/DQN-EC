import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np

# Create figure and axes
fig, ax = plt.subplots(figsize=(12, 8))

# Initial parameters
n_nodes = 20
n_steps = 50

# Create initial fully connected graph
G = nx.complete_graph(n_nodes)

# Initialize positions - two clusters
np.random.seed(123456)
pos = {}
for i in range(n_nodes):
    if i < n_nodes // 2:
        pos[i] = np.array([np.random.uniform(-0.4, 0), np.random.uniform(-0.4, 0.4)])
    else:
        pos[i] = np.array([np.random.uniform(0, 0.4), np.random.uniform(-0.4, 0.4)])

# Target positions for final state
target_pos = {}
for i in range(n_nodes):
    if i < n_nodes // 2:
        pos[i] = np.array([np.random.uniform(-0.4, 0), np.random.uniform(-0.4, 0.4)])
        target_pos[i] = np.array([np.random.uniform(-0.5, -0.1), np.random.uniform(-0.5, 0.5)])
    else:
        pos[i] = np.array([np.random.uniform(0, 0.4), np.random.uniform(-0.4, 0.4)])
        target_pos[i] = np.array([np.random.uniform(0.1, 0.5), np.random.uniform(-0.5, 0.5)])

# Get all edges and shuffle them for removal
edges_to_remove = list(G.edges())
np.random.shuffle(edges_to_remove)

# Keep some edges within clusters and a few between clusters
edges_to_keep = []
for i in range(n_nodes // 2):
    for j in range(i + 1, n_nodes // 2):
        if np.random.random() < 0.3:  # 30% of intra-cluster edges
            edges_to_keep.append((i, j))

for i in range(n_nodes // 2, n_nodes):
    for j in range(i + 1, n_nodes):
        if np.random.random() < 0.3:  # 30% of intra-cluster edges
            edges_to_keep.append((i, j))

# Add a few inter-cluster edges
for i in range(n_nodes // 2):
    for j in range(n_nodes // 2, n_nodes):
        if np.random.random() < 0.05:  # 5% of inter-cluster edges
            edges_to_keep.append((i, j))

# Remove edges that are not in keep list
edges_to_remove = [e for e in edges_to_remove if e not in edges_to_keep and (e[1], e[0]) not in edges_to_keep]

# Calculate how many edges to remove per step
edges_per_step = len(edges_to_remove) // n_steps + 1

# Metrics interpolation
episode_reward_start = -76
episode_reward_end = -65
agent_reward_start = -6.5
agent_reward_end = -5.4


def update(frame):
    ax.clear()

    # Remove edges for this frame
    edges_removed = min(frame * edges_per_step, len(edges_to_remove))
    for i in range(edges_removed):
        if i < len(edges_to_remove) and G.has_edge(*edges_to_remove[i]):
            G.remove_edge(*edges_to_remove[i])

    # Remove nodes with degree 0 (isolated nodes)
    isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
    G.remove_nodes_from(isolated_nodes)

    # Update positions gradually
    alpha = frame / n_steps
    current_pos = {}
    for node in G.nodes():  # Only iterate over remaining nodes
        if node in pos:
            current_pos[node] = pos[node] * (1 - alpha) + target_pos[node] * alpha

    # Calculate metrics
    episode_reward = episode_reward_start + (episode_reward_end - episode_reward_start) * alpha
    agent_reward = agent_reward_start + (agent_reward_end - agent_reward_start) * alpha

    # Draw graph
    nx.draw_networkx_nodes(G, current_pos, node_color='lightblue',
                           node_size=300, ax=ax)
    nx.draw_networkx_edges(G, current_pos, alpha=0.3, ax=ax)
    nx.draw_networkx_labels(G, current_pos, font_size=8, ax=ax)

    # Set axis properties
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 1.0)
    ax.axis('off')

    # Create metrics table
    table_data = [
        [f'{episode_reward:.2f}', f'{agent_reward:.2f}']
    ]

    # Position table in the upper part of the figure
    table = ax.table(cellText=table_data,
                     colLabels=['Episode Reward', 'Agent Reward'],
                     cellLoc='center',
                     loc='top',
                     bbox=[0.2, 0.92, 0.6, 0.08])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style the table
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(1, i)].set_facecolor('#E8F5E9')


# Create animation
anim = animation.FuncAnimation(fig, update, frames=n_steps, interval=200, repeat=False)

plt.tight_layout()
plt.show()