"""
Hypergraph Construction Framework
"""

from __future__ import annotations

import itertools as it
import math
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


class PyGHypergraphBuilder:
    """
    Build hypergraphs from PyTorch Geometric Data objects.
    
    Takes multiple PyG Data objects as components and creates a unified
    hypergraph with preserved node features and new hyperedge connections.
    """
    
    def __init__(self):
        self.metadata = {}
    
    @classmethod
    def build_strict(
        cls,
        data_list: List[Data],
        s: int,
        hub_component: int = 0,
    ) -> Data:
        """
        Build strict-bridging hypergraph from PyG Data objects using star batching.
        
        Parameters
        ----------
        data_list : List[Data]
            List of PyG Data objects, one per component
        s : int
            Maximum hyperedge size (≥2)
        hub_component : int
            Component index to use as hub for star batching
            
        Returns
        -------
        Data
            Unified PyG Data with hyperedges and preserved features
        """
        if s < 2:
            raise ValueError("s must be ≥ 2 to allow bridging hyperedges")
        if not data_list:
            raise ValueError("data_list cannot be empty")
            
        builder = cls()
        
        # Prepare merged data and metadata
        merged_data, metadata = builder._merge_components(data_list)
        builder.metadata = metadata
        m = metadata["m"]
        
        # Build bridging hyperedges using star batching
        bridging_edges = builder._star_batching(
            metadata["reps_global"], hub_component, m, s
        )
        
        # Convert hyperedges to edge pairs and add to graph
        result = builder._add_hyperedges_to_data(
            merged_data, bridging_edges, metadata
        )
        
        # Store algorithm info
        result.hypergraph_info = {
            "algorithm": "strict",
            "s": s,
            "hub_component": hub_component,
            "num_components": m,
            "num_bridging_edges": len(bridging_edges),
        }
        
        return result
    
    @classmethod
    def build_relaxed(
        cls,
        data_list: List[Data],
        s: int,
        rho: Optional[float] = None,
        feasible_override: Optional[np.ndarray] = None,
        load_cap: Optional[int] = None,
        anchor_strategy: str = "max_feasible",
        force_connect: bool = False,
    ) -> Data:
        """
        Build relaxed bridging hypergraph from PyG Data objects.
        
        Parameters
        ----------
        data_list : List[Data]
            List of PyG Data objects, one per component
        s : int
            Maximum hyperedge size (≥2)
        rho : Optional[float]
            Feasibility radius (components within 2*rho are feasible)
        feasible_override : Optional[np.ndarray]
            Explicit (m x m) boolean feasibility matrix
        load_cap : Optional[int]
            Maximum load per vertex
        anchor_strategy : str
            Strategy for selecting anchor components
        force_connect : bool
            Force connection even if feasibility graph is disconnected
            
        Returns
        -------
        Data
            Unified PyG Data with hyperedges and preserved features
        """
        if s < 2:
            raise ValueError("s must be ≥ 2 to allow bridging hyperedges")
        if not data_list:
            raise ValueError("data_list cannot be empty")
            
        builder = cls()
        
        # Prepare merged data and metadata
        merged_data, metadata = builder._merge_components(data_list)
        builder.metadata = metadata
        m = metadata["m"]
        
        # Build feasibility matrix
        F_comp = builder._build_feasibility_matrix(
            merged_data.pos.numpy() if merged_data.pos is not None else None,
            metadata["comp_vertices"],
            rho,
            feasible_override
        )
        
        # Initialize DSU and tracking
        dsu = builder._DSU(m)
        vertex_load = {}
        bridging_edges = []
        
        # Build bridging hyperedges iteratively
        while dsu.num_roots() > 1:
            # Find feasible adjacencies
            adj = builder._compute_root_adjacencies(dsu, F_comp, m)
            
            if not adj or all(len(nbrs) == 0 for nbrs in adj.values()):
                if not force_connect:
                    raise RuntimeError(
                        "Feasibility graph is disconnected under given constraints"
                    )
                # Force connection between nearest pair
                bridging = builder._force_nearest_connection(
                    dsu, metadata, vertex_load, merged_data.pos.numpy()
                )
                if bridging:
                    bridging_edges.append(bridging)
                continue
            
            # Select anchor and neighbors
            anchor = builder._select_anchor(adj, anchor_strategy)
            neighbors = builder._choose_neighbors(
                anchor, s - 1, adj, dsu, F_comp, metadata
            )
            
            # Build hyperedge
            chosen_roots = [anchor] + neighbors
            hyperedge = builder._form_hyperedge(
                chosen_roots,
                dsu,
                metadata,
                vertex_load,
                load_cap,
                merged_data.pos.numpy() if merged_data.pos is not None else None
            )
            
            if hyperedge and len(hyperedge) >= 2:
                bridging_edges.append(hyperedge)
                dsu.union_many(chosen_roots)
        
        # Convert hyperedges to edge pairs and add to graph
        result = builder._add_hyperedges_to_data(
            merged_data, bridging_edges, metadata
        )
        
        # Store algorithm info
        result.hypergraph_info = {
            "algorithm": "relaxed",
            "s": s,
            "rho": rho,
            "load_cap": load_cap,
            "num_components": m,
            "num_bridging_edges": len(bridging_edges),
        }
        
        return result
    
    # ========================= Helper Methods =========================
    
    def _merge_components(self, data_list: List[Data]) -> Tuple[Data, Dict]:
        """
        Merge multiple PyG Data objects into one, tracking component membership.
        
        Returns merged Data and metadata dictionary.
        """
        # Initialize tracking
        comp_vertices = []  # Global vertex indices for each component
        comp_of = []  # Component ID for each vertex
        current_idx = 0
        
        # Collect all features
        all_x = []
        all_pos = []
        all_edges = []
        all_edge_attr = []
        
        # Process each component
        for cid, data in enumerate(data_list):
            n = data.num_nodes
            
            # Track vertex indices
            vertex_ids = list(range(current_idx, current_idx + n))
            comp_vertices.append(vertex_ids)
            comp_of.extend([cid] * n)
            
            # Collect node features
            if data.x is not None:
                all_x.append(data.x)
            else:
                # Create dummy features if not present
                all_x.append(torch.zeros((n, 1), dtype=torch.float32))
            
            # Collect positions
            if data.pos is not None:
                all_pos.append(data.pos)
            else:
                # Generate random positions if not present
                all_pos.append(torch.randn((n, 2), dtype=torch.float32))
            
            # Collect edges (adjust indices to global)
            if data.edge_index is not None and data.edge_index.shape[1] > 0:
                adjusted_edges = data.edge_index + current_idx
                all_edges.append(adjusted_edges)
                
                # Collect edge attributes
                if data.edge_attr is not None:
                    all_edge_attr.append(data.edge_attr)
                else:
                    # Compute euclidean distances as edge attributes
                    pos = all_pos[-1]
                    src, dst = data.edge_index
                    distances = torch.norm(pos[src] - pos[dst], dim=1, keepdim=True)
                    all_edge_attr.append(distances)
            
            current_idx += n
        
        # Create merged Data object
        merged = Data()
        
        # Merge node features
        merged.x = torch.cat(all_x, dim=0) if all_x else torch.zeros((current_idx, 1))
        
        # Merge positions
        merged.pos = torch.cat(all_pos, dim=0) if all_pos else torch.zeros((current_idx, 2))
        
        # Merge edges
        if all_edges:
            merged.edge_index = torch.cat(all_edges, dim=1)
            merged.edge_attr = torch.cat(all_edge_attr, dim=0) if all_edge_attr else None
        else:
            merged.edge_index = torch.empty((2, 0), dtype=torch.long)
            merged.edge_attr = None
        
        # Add component labels
        merged.comp = torch.tensor(comp_of, dtype=torch.long)
        
        # Find representatives (closest to centroid for each component)
        reps_global = []
        for cid, verts in enumerate(comp_vertices):
            if not verts:
                reps_global.append(-1)
            else:
                pts = merged.pos[verts].numpy()
                centroid = pts.mean(axis=0)
                d = np.linalg.norm(pts - centroid, axis=1)
                reps_global.append(verts[int(np.argmin(d))])
        
        # Create metadata
        metadata = {
            "comp_vertices": comp_vertices,
            "comp_of": comp_of,
            "reps_global": reps_global,
            "m": len(data_list),
            "original_edge_count": merged.edge_index.shape[1],
        }
        
        return merged, metadata
    
    def _add_hyperedges_to_data(
        self,
        data: Data,
        hyperedges: List[frozenset[int]],
        metadata: Dict
    ) -> Data:
        """
        Add hyperedges to the PyG Data object as directed edge pairs.
        
        Each hyperedge {a, b, c} becomes edges: (a,b), (b,a), (b,c), (c,b), (a,c), (c,a)
        """
        # Convert hyperedges to edge pairs (fully connected within each hyperedge)
        new_edges = []
        new_edge_attr = []
        
        for he in hyperedges:
            vertices = sorted(list(he))
            # Create all pairwise connections within hyperedge
            for u, v in it.combinations(vertices, 2):
                # Add both directions
                new_edges.extend([[u, v], [v, u]])
                
                # Compute euclidean distance as edge attribute
                if data.pos is not None:
                    dist = torch.norm(data.pos[u] - data.pos[v]).item()
                    new_edge_attr.extend([dist, dist])  # Same distance for both directions
        
        if new_edges:
            new_edge_tensor = torch.tensor(new_edges, dtype=torch.long).T
            
            # Combine with existing edges
            if data.edge_index is not None and data.edge_index.shape[1] > 0:
                data.edge_index = torch.cat([data.edge_index, new_edge_tensor], dim=1)
                
                # Add edge type attribute (0 for original, 1 for hyperedge)
                n_orig = metadata["original_edge_count"]
                n_new = new_edge_tensor.shape[1]
                edge_type = torch.cat([
                    torch.zeros(n_orig, dtype=torch.long),
                    torch.ones(n_new, dtype=torch.long)
                ])
                data.edge_type = edge_type
                
                # Combine edge attributes
                if new_edge_attr:
                    new_attr_tensor = torch.tensor(new_edge_attr, dtype=torch.float32).unsqueeze(1)
                    if data.edge_attr is not None:
                        data.edge_attr = torch.cat([data.edge_attr, new_attr_tensor], dim=0)
                    else:
                        # Create dummy attributes for original edges
                        orig_attr = torch.zeros((n_orig, 1), dtype=torch.float32)
                        data.edge_attr = torch.cat([orig_attr, new_attr_tensor], dim=0)
            else:
                data.edge_index = new_edge_tensor
                data.edge_type = torch.ones(new_edge_tensor.shape[1], dtype=torch.long)
                if new_edge_attr:
                    data.edge_attr = torch.tensor(new_edge_attr, dtype=torch.float32).unsqueeze(1)
        
        # Store hyperedge information
        data.hyperedges = [list(he) for he in hyperedges]
        
        return data
    
    @staticmethod
    def _star_batching(
        reps_global: List[int], hub: int, m: int, s: int
    ) -> List[frozenset[int]]:
        """Create bridging hyperedges using star batching."""
        bridging_edges = []
        others = [j for j in range(m) if j != hub]
        block_size = s - 1
        
        for start in range(0, len(others), block_size):
            block = others[start : start + block_size]
            he = {reps_global[hub]} | {reps_global[j] for j in block}
            if len(he) >= 2:
                bridging_edges.append(frozenset(he))
        
        return bridging_edges
    
    @staticmethod
    def _build_feasibility_matrix(
        P_all: Optional[np.ndarray],
        comp_vertices: List[List[int]],
        rho: Optional[float],
        override: Optional[np.ndarray],
    ) -> np.ndarray:
        """Build component feasibility matrix."""
        m = len(comp_vertices)
        
        if override is not None:
            F = override.astype(bool).copy()
            np.fill_diagonal(F, False)
            return F
        
        F = np.zeros((m, m), dtype=bool)
        if rho is None or P_all is None:
            F[:] = True
            np.fill_diagonal(F, False)
        else:
            threshold = 2.0 * rho
            for i in range(m):
                for j in range(i + 1, m):
                    if not comp_vertices[i] or not comp_vertices[j]:
                        continue
                    A = P_all[comp_vertices[i]]
                    B = P_all[comp_vertices[j]]
                    diffs = A[:, None, :] - B[None, :, :]
                    min_dist = np.sqrt((diffs**2).sum(axis=2).min())
                    ok = min_dist <= threshold
                    F[i, j] = F[j, i] = ok
        
        return F
    
    @staticmethod
    def _compute_root_adjacencies(dsu, F_comp: np.ndarray, m: int) -> Dict:
        """Compute feasible adjacencies between DSU roots."""
        roots = dsu.all_roots()
        adj = {r: set() for r in roots}
        
        for i in range(m):
            ri = dsu.find(i)
            for j in range(i + 1, m):
                rj = dsu.find(j)
                if ri != rj and F_comp[i, j]:
                    adj[ri].add(rj)
                    adj[rj].add(ri)
        
        return {r: nbrs for r, nbrs in adj.items() if r in roots}
    
    @staticmethod
    def _select_anchor(adj: Dict, strategy: str) -> int:
        """Select anchor root based on strategy."""
        if not adj:
            return -1
        if strategy == "max_feasible":
            return max(adj.keys(), key=lambda r: len(adj[r]))
        else:
            return list(adj.keys())[0]
    
    @staticmethod
    def _choose_neighbors(
        anchor: int,
        k: int,
        adj: Dict,
        dsu,
        F_comp: np.ndarray,
        metadata: Dict,
    ) -> List[int]:
        """Choose k neighbor roots for the anchor."""
        nbrs = list(adj.get(anchor, set()))
        if not nbrs or k <= 0:
            return []
        return nbrs[:k]
    
    def _form_hyperedge(
        self,
        chosen_roots: List[int],
        dsu,
        metadata: Dict,
        vertex_load: Dict,
        load_cap: Optional[int],
        P_all: Optional[np.ndarray],
    ) -> Optional[frozenset[int]]:
        """Form a hyperedge from chosen roots."""
        comp_vertices = metadata["comp_vertices"]
        comp_of = metadata["comp_of"]
        
        # Get components for each root
        root_to_comps = {}
        for c in range(metadata["m"]):
            r = dsu.find(c)
            if r in chosen_roots:
                root_to_comps.setdefault(r, []).append(c)
        
        # Select one vertex per root (respecting load cap)
        used = []
        seen_components = set()
        
        for r in chosen_roots:
            best_v = None
            best_score = float("inf")
            
            for c in root_to_comps.get(r, []):
                if c in seen_components:
                    continue
                verts = comp_vertices[c]
                if not verts:
                    continue
                
                # Find best vertex (closest to centroid, respecting load)
                if P_all is not None:
                    pts = P_all[verts]
                    centroid = pts.mean(axis=0)
                    dists = np.linalg.norm(pts - centroid, axis=1)
                    
                    for idx in np.argsort(dists):
                        v = verts[idx]
                        load = vertex_load.get(v, 0)
                        if load_cap is not None and load >= load_cap:
                            continue
                        if dists[idx] < best_score:
                            best_score = dists[idx]
                            best_v = v
                        break
                else:
                    # Just pick first available vertex
                    for v in verts:
                        load = vertex_load.get(v, 0)
                        if load_cap is None or load < load_cap:
                            best_v = v
                            break
            
            if best_v is not None:
                used.append(best_v)
                seen_components.add(comp_of[best_v])
                vertex_load[best_v] = vertex_load.get(best_v, 0) + 1
        
        if len(used) >= 2:
            return frozenset(used)
        return None
    
    def _force_nearest_connection(
        self, dsu, metadata: Dict, vertex_load: Dict, P_all: Optional[np.ndarray]
    ) -> Optional[frozenset[int]]:
        """Force connection between nearest pair of roots."""
        roots = dsu.all_roots()
        if len(roots) <= 1:
            return None
        
        if P_all is None:
            # Just connect first two components
            r1, r2 = roots[:2]
            v1 = metadata["reps_global"][r1]
            v2 = metadata["reps_global"][r2]
            dsu.union_many([r1, r2])
            return frozenset({v1, v2})
        
        # Find nearest pair of components
        comp_vertices = metadata["comp_vertices"]
        best_dist = float("inf")
        best_pair = None
        
        for i, ri in enumerate(roots):
            for rj in roots[i + 1 :]:
                # Find min distance between components
                for ci in range(metadata["m"]):
                    if dsu.find(ci) != ri:
                        continue
                    for cj in range(metadata["m"]):
                        if dsu.find(cj) != rj:
                            continue
                        if not comp_vertices[ci] or not comp_vertices[cj]:
                            continue
                        
                        A = P_all[comp_vertices[ci]]
                        B = P_all[comp_vertices[cj]]
                        diffs = A[:, None, :] - B[None, :, :]
                        min_dist = np.sqrt((diffs**2).sum(axis=2).min())
                        
                        if min_dist < best_dist:
                            best_dist = min_dist
                            idx = np.unravel_index(
                                np.argmin((diffs**2).sum(axis=2)), (len(A), len(B))
                            )
                            best_pair = (
                                comp_vertices[ci][idx[0]],
                                comp_vertices[cj][idx[1]],
                            )
        
        if best_pair:
            for v in best_pair:
                vertex_load[v] = vertex_load.get(v, 0) + 1
            dsu.union_many([dsu.find(metadata["comp_of"][v]) for v in best_pair])
            return frozenset(best_pair)
        
        return None
    
    # ========================= DSU Helper Class =========================
    
    class _DSU:
        """Disjoint Set Union data structure for component merging."""
        
        def __init__(self, n: int):
            self.parent = list(range(n))
            self.size = [1] * n
        
        def find(self, x: int) -> int:
            """Find root with path compression."""
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union_many(self, roots: List[int]) -> int:
            """Union multiple roots, return new root."""
            roots = list({self.find(r) for r in roots})
            if not roots:
                return -1
            roots.sort(key=lambda r: -self.size[r])
            base = roots[0]
            for r in roots[1:]:
                r = self.find(r)
                if r != base:
                    self.parent[r] = base
                    self.size[base] += self.size[r]
            return base
        
        def num_roots(self) -> int:
            """Count number of distinct roots."""
            return len(set(self.find(i) for i in range(len(self.parent))))
        
        def all_roots(self) -> List[int]:
            """Get all distinct roots."""
            return sorted(set(self.find(i) for i in range(len(self.parent))))


# ========================= Utility Functions =========================


def create_sample_components(
    n_components: int = 4,
    nodes_per_comp: int = 10,
    feature_dim: int = 8,
    seed: int = 42
) -> List[Data]:
    """
    Create sample PyG Data objects for testing.
    
    Parameters
    ----------
    n_components : int
        Number of components to create
    nodes_per_comp : int
        Number of nodes per component
    feature_dim : int
        Dimension of node features
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    List[Data]
        List of PyG Data objects
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    data_list = []
    
    for i in range(n_components):
        # Create node features
        x = torch.randn((nodes_per_comp, feature_dim))
        
        # Create positions in 2D space
        center = torch.tensor([i * 3.0, (i % 2) * 3.0])
        pos = center + torch.randn((nodes_per_comp, 2))
        
        # Create edges (chain + some random connections)
        edges = []
        # Chain backbone
        for j in range(nodes_per_comp - 1):
            edges.extend([[j, j + 1], [j + 1, j]])
        
        # Add some random edges
        for _ in range(nodes_per_comp // 3):
            a, b = np.random.choice(nodes_per_comp, 2, replace=False)
            if a != b:
                edges.extend([[a, b], [b, a]])
        
        edge_index = torch.tensor(edges, dtype=torch.long).T
        
        # Compute edge attributes (distances)
        src, dst = edge_index
        edge_attr = torch.norm(pos[src] - pos[dst], dim=1, keepdim=True)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos
        )
        
        data_list.append(data)
    
    return data_list


def visualize_hypergraph(
    data: Data,
    title: str = "Hypergraph Visualization",
    figsize: Tuple[float, float] = (12, 10),
    save_path: Optional[str] = None,
    show_hyperedge_hulls: bool = True,
    node_size: int = 100,
    show_legend: bool = True
):
    """
    Visualize a PyG hypergraph with components and hyperedges.
    
    Parameters
    ----------
    data : Data
        PyG Data object with hypergraph
    title : str
        Plot title
    figsize : Tuple[float, float]
        Figure size
    save_path : Optional[str]
        Path to save figure
    show_hyperedge_hulls : bool
        Whether to show convex hulls around hyperedges
    node_size : int
        Size of nodes in plot
    show_legend : bool
        Whether to show legend
    """
    from scipy.spatial import ConvexHull
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Get positions and component labels
    if data.pos is not None:
        pos = data.pos.numpy()
    else:
        # Create layout using networkx if no positions
        G = nx.Graph()
        G.add_edges_from(data.edge_index.T.numpy())
        pos = nx.spring_layout(G)
        pos = np.array([pos[i] for i in range(data.num_nodes)])
    
    comp_labels = data.comp.numpy() if hasattr(data, 'comp') else np.zeros(data.num_nodes)
    n_components = len(np.unique(comp_labels))
    
    # Color map for components
    colors = plt.cm.tab10(np.linspace(0, 1, n_components))
    
    # ========== Left plot: Original structure ==========
    ax1.set_title(f"{title}\n(Original + Component Structure)", fontsize=11, fontweight='bold')
    
    # Plot nodes by component
    for comp_id in range(n_components):
        mask = comp_labels == comp_id
        ax1.scatter(pos[mask, 0], pos[mask, 1], 
                   c=[colors[comp_id]], s=node_size, 
                   label=f'Component {comp_id}', alpha=0.8, edgecolors='black', linewidth=1)
    
    # Plot original edges only
    if hasattr(data, 'edge_type'):
        edge_mask = data.edge_type == 0
        edges_to_plot = data.edge_index[:, edge_mask]
    else:
        edges_to_plot = data.edge_index
    
    for i in range(edges_to_plot.shape[1]):
        src, dst = edges_to_plot[:, i]
        ax1.plot([pos[src, 0], pos[dst, 0]], 
                [pos[src, 1], pos[dst, 1]], 
                'k-', alpha=0.3, linewidth=0.5)
    
    # ========== Right plot: Hyperedge structure ==========
    ax2.set_title(f"{title}\n(Hyperedge Connections)", fontsize=11, fontweight='bold')
    
    # Plot nodes by component
    for comp_id in range(n_components):
        mask = comp_labels == comp_id
        ax2.scatter(pos[mask, 0], pos[mask, 1], 
                   c=[colors[comp_id]], s=node_size, 
                   alpha=0.8, edgecolors='black', linewidth=1)
    
    # Plot original edges (faint)
    if hasattr(data, 'edge_type'):
        edge_mask = data.edge_type == 0
        edges_orig = data.edge_index[:, edge_mask]
        for i in range(edges_orig.shape[1]):
            src, dst = edges_orig[:, i]
            ax2.plot([pos[src, 0], pos[dst, 0]], 
                    [pos[src, 1], pos[dst, 1]], 
                    'gray', alpha=0.1, linewidth=0.5)
    
    # Plot hyperedges
    if hasattr(data, 'hyperedges'):
        hyperedge_colors = plt.cm.Set2(np.linspace(0, 1, len(data.hyperedges)))
        
        for he_idx, hyperedge in enumerate(data.hyperedges):
            if len(hyperedge) < 2:
                continue
            
            he_pos = pos[hyperedge]
            
            # Draw convex hull around hyperedge
            if show_hyperedge_hulls and len(hyperedge) >= 3:
                try:
                    hull = ConvexHull(he_pos)
                    hull_points = he_pos[hull.vertices]
                    poly = Polygon(hull_points, alpha=0.15, 
                                 facecolor=hyperedge_colors[he_idx], 
                                 edgecolor=hyperedge_colors[he_idx], linewidth=2)
                    ax2.add_patch(poly)
                except:
                    pass  # Skip if hull computation fails
            
            # Draw edges within hyperedge
            for i in range(len(hyperedge)):
                for j in range(i+1, len(hyperedge)):
                    v1, v2 = hyperedge[i], hyperedge[j]
                    ax2.plot([pos[v1, 0], pos[v2, 0]], 
                            [pos[v1, 1], pos[v2, 1]], 
                            color=hyperedge_colors[he_idx], 
                            alpha=0.7, linewidth=2, linestyle='--')
            
            # Mark hyperedge nodes
            ax2.scatter(he_pos[:, 0], he_pos[:, 1], 
                       s=node_size*1.5, facecolors='none', 
                       edgecolors=hyperedge_colors[he_idx], linewidth=2)
    
    # Plot hyperedge connections (if edge_type available)
    elif hasattr(data, 'edge_type'):
        edge_mask = data.edge_type == 1
        edges_hyper = data.edge_index[:, edge_mask]
        for i in range(edges_hyper.shape[1]):
            src, dst = edges_hyper[:, i]
            ax2.plot([pos[src, 0], pos[dst, 0]], 
                    [pos[src, 1], pos[dst, 1]], 
                    'r--', alpha=0.5, linewidth=1.5)
    
    # Formatting
    for ax in [ax1, ax2]:
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
    
    if show_legend:
        ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    # Add info text
    if hasattr(data, 'hypergraph_info'):
        info = data.hypergraph_info
        info_text = f"Algorithm: {info['algorithm']}, s={info['s']}, "
        info_text += f"Components: {info['num_components']}, "
        info_text += f"Bridging edges: {info['num_bridging_edges']}"
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved visualization to {save_path}")
    else:
        plt.show()
    
    return fig


def analyze_hypergraph_connectivity(data: Data) -> Dict:
    """
    Analyze connectivity properties of the hypergraph.
    
    Parameters
    ----------
    data : Data
        PyG Data object with hypergraph
        
    Returns
    -------
    Dict
        Analysis results including connectivity metrics
    """
    import networkx as nx
    
    # Create NetworkX graph for analysis
    G = nx.Graph()
    edges = data.edge_index.T.numpy()
    G.add_edges_from(edges)
    
    # Basic connectivity
    is_connected = nx.is_connected(G)
    n_components = nx.number_connected_components(G)
    
    # Component analysis
    comp_labels = data.comp.numpy() if hasattr(data, 'comp') else np.zeros(data.num_nodes)
    n_original_components = len(np.unique(comp_labels))
    
    # Edge type analysis
    if hasattr(data, 'edge_type'):
        n_original_edges = (data.edge_type == 0).sum().item() // 2  # Undirected
        n_hyperedge_connections = (data.edge_type == 1).sum().item() // 2
    else:
        n_original_edges = data.edge_index.shape[1] // 2
        n_hyperedge_connections = 0
    
    # Hyperedge analysis
    if hasattr(data, 'hyperedges'):
        hyperedge_sizes = [len(he) for he in data.hyperedges]
        avg_hyperedge_size = np.mean(hyperedge_sizes) if hyperedge_sizes else 0
        max_hyperedge_size = max(hyperedge_sizes) if hyperedge_sizes else 0
    else:
        hyperedge_sizes = []
        avg_hyperedge_size = 0
        max_hyperedge_size = 0
    
    # Degree analysis
    degrees = dict(G.degree())
    avg_degree = np.mean(list(degrees.values()))
    max_degree = max(degrees.values()) if degrees else 0
    
    # Shortest path analysis (sample if graph is large)
    if G.number_of_nodes() <= 100 and is_connected:
        avg_path_length = nx.average_shortest_path_length(G)
        diameter = nx.diameter(G)
    else:
        avg_path_length = -1
        diameter = -1
    
    return {
        'is_connected': is_connected,
        'n_connected_components': n_components,
        'n_original_components': n_original_components,
        'n_nodes': data.num_nodes,
        'n_edges': data.num_edges // 2,  # Undirected
        'n_original_edges': n_original_edges,
        'n_hyperedge_connections': n_hyperedge_connections,
        'n_hyperedges': len(data.hyperedges) if hasattr(data, 'hyperedges') else 0,
        'hyperedge_sizes': hyperedge_sizes,
        'avg_hyperedge_size': avg_hyperedge_size,
        'max_hyperedge_size': max_hyperedge_size,
        'avg_degree': avg_degree,
        'max_degree': max_degree,
        'avg_shortest_path': avg_path_length,
        'diameter': diameter,
    }


def demo():
    """Demonstrate PyG hypergraph construction with visualization."""
    print("=" * 60)
    print("PyTorch Geometric Hypergraph Construction Demo")
    print("=" * 60)
    
    # Create sample components with clearer spatial separation
    print("\n1. Creating sample PyG Data components...")
    components = create_sample_components(
        n_components=5,
        nodes_per_comp=8,
        feature_dim=16,
        seed=42
    )
    print(f"   Created {len(components)} components")
    for i, data in enumerate(components):
        print(f"   Component {i}: {data.num_nodes} nodes, {data.num_edges} edges")
    
    # Build strict hypergraph
    print("\n2. Building STRICT hypergraph...")
    hg_strict = PyGHypergraphBuilder.build_strict(
        data_list=components,
        s=3,
        hub_component=1
    )
    print(f"   Total nodes: {hg_strict.num_nodes}")
    print(f"   Total edges: {hg_strict.num_edges}")
    print(f"   Hyperedges created: {len(hg_strict.hyperedges)}")
    
    # Analyze connectivity
    analysis_strict = analyze_hypergraph_connectivity(hg_strict)
    print(f"   Connected: {analysis_strict['is_connected']}")
    print(f"   Average hyperedge size: {analysis_strict['avg_hyperedge_size']:.2f}")
    print(f"   Average node degree: {analysis_strict['avg_degree']:.2f}")
    
    # Build relaxed hypergraph (no constraints)
    print("\n3. Building RELAXED hypergraph (no constraints)...")
    hg_relaxed = PyGHypergraphBuilder.build_relaxed(
        data_list=components,
        s=3,
        rho=None
    )
    print(f"   Total nodes: {hg_relaxed.num_nodes}")
    print(f"   Total edges: {hg_relaxed.num_edges}")
    print(f"   Hyperedges created: {len(hg_relaxed.hyperedges)}")
    
    # Analyze connectivity
    analysis_relaxed = analyze_hypergraph_connectivity(hg_relaxed)
    print(f"   Connected: {analysis_relaxed['is_connected']}")
    print(f"   Average hyperedge size: {analysis_relaxed['avg_hyperedge_size']:.2f}")
    print(f"   Average node degree: {analysis_relaxed['avg_degree']:.2f}")
    
    # Build relaxed hypergraph (with radius constraint)
    print("\n4. Building RELAXED hypergraph (with radius constraint)...")
    hg_constrained = PyGHypergraphBuilder.build_relaxed(
        data_list=components,
        s=4,
        rho=0.5,
        force_connect=True
    )
    print(f"   Total nodes: {hg_constrained.num_nodes}")
    print(f"   Total edges: {hg_constrained.num_edges}")
    print(f"   Hyperedges created: {len(hg_constrained.hyperedges)}")
    
    # Analyze connectivity
    analysis_constrained = analyze_hypergraph_connectivity(hg_constrained)
    print(f"   Connected: {analysis_constrained['is_connected']}")
    print(f"   Average hyperedge size: {analysis_constrained['avg_hyperedge_size']:.2f}")
    print(f"   Average node degree: {analysis_constrained['avg_degree']:.2f}")
    
    # Show feature preservation
    print("\n5. Verifying feature preservation...")
    print(f"   Node features shape: {hg_strict.x.shape}")
    print(f"   Node positions shape: {hg_strict.pos.shape}")
    print(f"   Edge attributes shape: {hg_strict.edge_attr.shape if hg_strict.edge_attr is not None else 'None'}")
    print(f"   Component labels shape: {hg_strict.comp.shape}")
    
    # Show edge types
    if hasattr(hg_strict, 'edge_type'):
        n_original = (hg_strict.edge_type == 0).sum().item()
        n_hyperedge = (hg_strict.edge_type == 1).sum().item()
        print(f"   Original edges: {n_original}")
        print(f"   Hyperedge connections: {n_hyperedge}")
    
    # Visualize the hypergraphs
    print("\n6. Generating visualizations...")
    
    visualize_hypergraph(
        hg_strict,
        title="Strict Hypergraph",
        save_path="hypergraph_strict_pyg.png",
        show_hyperedge_hulls=True
    )
    
    visualize_hypergraph(
        hg_relaxed,
        title="Relaxed Hypergraph (No Constraints)",
        save_path="hypergraph_relaxed_pyg.png",
        show_hyperedge_hulls=True
    )
    
    visualize_hypergraph(
        hg_constrained,
        title="Relaxed Hypergraph (ρ=2.0)",
        save_path="hypergraph_constrained_pyg.png",
        show_hyperedge_hulls=True
    )
    
    # Print connectivity comparison
    print("\n7. Connectivity Comparison:")
    print("   " + "-" * 50)
    print(f"   {'Metric':<30} {'Strict':<10} {'Relaxed':<10} {'Constrained':<10}")
    print("   " + "-" * 50)
    print(f"   {'Connected':<30} {str(analysis_strict['is_connected']):<10} {str(analysis_relaxed['is_connected']):<10} {str(analysis_constrained['is_connected']):<10}")
    print(f"   {'Hyperedges':<30} {analysis_strict['n_hyperedges']:<10} {analysis_relaxed['n_hyperedges']:<10} {analysis_constrained['n_hyperedges']:<10}")
    print(f"   {'Avg Hyperedge Size':<30} {analysis_strict['avg_hyperedge_size']:<10.2f} {analysis_relaxed['avg_hyperedge_size']:<10.2f} {analysis_constrained['avg_hyperedge_size']:<10.2f}")
    print(f"   {'Avg Node Degree':<30} {analysis_strict['avg_degree']:<10.2f} {analysis_relaxed['avg_degree']:<10.2f} {analysis_constrained['avg_degree']:<10.2f}")
    print(f"   {'Max Node Degree':<30} {analysis_strict['max_degree']:<10} {analysis_relaxed['max_degree']:<10} {analysis_constrained['max_degree']:<10}")
    
    print("\n" + "=" * 60)
    print("Demo complete! Visualizations saved.")
    print("=" * 60)


if __name__ == "__main__":
    demo()


# Export main components
__all__ = [
    "PyGHypergraphBuilder",
    "create_sample_components",
    "visualize_hypergraph",
    "analyze_hypergraph_connectivity",
    "demo",
]