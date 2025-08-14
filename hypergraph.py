"""
Unified Hypergraph Construction Framework with PyTorch Geometric Support

A clean, elegant implementation for building strict and relaxed bridging hypergraphs
with comprehensive analysis, visualization, and PyG integration capabilities.

Quick Usage:
-----------
    # Generate demo components
    components, edges = generate_demo_components(n_components=5)
    
    # Build strict hypergraph (star batching)
    hg_strict = Hypergraph.build_strict(components, edges, s=3)
    
    # Build relaxed hypergraph (with constraints)
    hg_relaxed = Hypergraph.build_relaxed(components, edges, s=3, rho=2.0)
    
    # Analyze and visualize
    checks = hg_strict.check_properties()
    hg_strict.plot(title="My Hypergraph")
    
    # Convert to PyTorch Geometric (if available)
    hetero_data = hg_strict.to_pyg_hetero()
    data = hg_strict.to_pyg_data()
"""

from __future__ import annotations

import itertools as it
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

# Optional PyTorch/PyG imports - gracefully handle if not available
try:
    import torch
    from torch import Tensor
    from torch_geometric.data import Data, HeteroData

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Tensor = None
    Data = None
    HeteroData = None
    TORCH_AVAILABLE = False


class Hypergraph:
    """
    Unified hypergraph builder with strict and relaxed bridging algorithms.

    This class provides a comprehensive framework for constructing hypergraphs
    from multi-component graphs, with support for both strict bridging (star
    batching) and relaxed bridging (Algorithm-1 style with feasibility constraints).

    Attributes
    ----------
    vertices : List[int]
        Global vertex indices [0..N-1]
    hyperedges : List[frozenset[int]]
        List of hyperedges, each a frozenset of vertex indices
    metadata : Dict
        Construction metadata including positions, component info, etc.
    """

    def __init__(
        self,
        vertices: List[int],
        hyperedges: List[frozenset[int]],
        metadata: Optional[Dict] = None,
    ):
        """Initialize a Hypergraph instance."""
        self.vertices = vertices
        self.hyperedges = hyperedges
        self.metadata = metadata or {}
        self._two_section_cache = None

    # ========================= Builder Methods =========================

    @classmethod
    def build_strict(
        cls,
        components_pts: List[np.ndarray],
        edges_per_component: List[Iterable[Tuple[int, int]]],
        s: int,
        hub_component: int = 0,
    ) -> Hypergraph:
        """
        Build a strict-bridging hypergraph using star batching.

        Parameters
        ----------
        components_pts : List[np.ndarray]
            List of point arrays, one per component. Each array shape: (n_i, 2)
        edges_per_component : List[Iterable[Tuple[int, int]]]
            Local edge indices for each component
        s : int
            Maximum hyperedge size (≥2)
        hub_component : int
            Component index to use as hub for star batching

        Returns
        -------
        Hypergraph
            Constructed hypergraph with strict bridging properties
        """
        if s < 2:
            raise ValueError("s must be ≥ 2 to allow bridging hyperedges")
        if len(components_pts) != len(edges_per_component):
            raise ValueError("components_pts and edges must have same length")

        # Process components and build metadata
        metadata = cls._prepare_metadata(components_pts, edges_per_component)
        m = metadata["m"]

        # Build hyperedges
        hyperedges = []

        # Add intra-component edges as 2-hyperedges
        for u, v in metadata["intra_edges_global"]:
            hyperedges.append(frozenset({u, v}))

        # Build bridging hyperedges using star batching
        bridging_edges = cls._star_batching(
            metadata["reps_global"], hub_component, m, s
        )
        hyperedges.extend(bridging_edges)

        # Update metadata
        metadata.update(
            {
                "bridging_edges": bridging_edges,
                "s": s,
                "hub_component": hub_component,
                "algorithm": "strict",
            }
        )

        return cls(
            vertices=list(range(len(metadata["P_all"]))),
            hyperedges=hyperedges,
            metadata=metadata,
        )

    @classmethod
    def build_relaxed(
        cls,
        components_pts: List[np.ndarray],
        edges_per_component: List[Iterable[Tuple[int, int]]],
        s: int,
        rho: Optional[float] = None,
        feasible_override: Optional[np.ndarray] = None,
        load_cap: Optional[int] = None,
        anchor_strategy: str = "max_feasible",
        force_connect: bool = False,
    ) -> Hypergraph:
        """
        Build a relaxed bridging hypergraph with feasibility constraints.

        Parameters
        ----------
        components_pts : List[np.ndarray]
            List of point arrays, one per component
        edges_per_component : List[Iterable[Tuple[int, int]]]
            Local edge indices for each component
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
        Hypergraph
            Constructed hypergraph with relaxed bridging
        """
        if s < 2:
            raise ValueError("s must be ≥ 2 to allow bridging hyperedges")

        # Prepare metadata
        metadata = cls._prepare_metadata(components_pts, edges_per_component)
        m = metadata["m"]

        # Build feasibility matrix
        F_comp = cls._build_feasibility_matrix(
            metadata["P_all"], metadata["comp_vertices"], rho, feasible_override
        )

        # Initialize DSU and tracking
        dsu = cls._DSU(m)
        vertex_load = {}
        hyperedges = []
        bridging_edges = []

        # Add intra-component edges
        for u, v in metadata["intra_edges_global"]:
            hyperedges.append(frozenset({u, v}))

        # Build bridging hyperedges iteratively
        while dsu.num_roots() > 1:
            # Find feasible adjacencies
            adj = cls._compute_root_adjacencies(dsu, F_comp, m)

            if not adj or all(len(nbrs) == 0 for nbrs in adj.values()):
                if not force_connect:
                    raise RuntimeError(
                        "Feasibility graph is disconnected under given constraints"
                    )
                # Force connection between nearest pair
                bridging = cls._force_nearest_connection(
                    dsu, metadata, vertex_load
                )
                if bridging:
                    hyperedges.append(bridging)
                    bridging_edges.append(bridging)
                continue

            # Select anchor and neighbors
            anchor = cls._select_anchor(adj, anchor_strategy)
            neighbors = cls._choose_neighbors(
                anchor, s - 1, adj, dsu, F_comp, metadata
            )

            # Build hyperedge
            chosen_roots = [anchor] + neighbors
            hyperedge = cls._form_hyperedge(
                chosen_roots,
                dsu,
                metadata,
                vertex_load,
                load_cap,
            )

            if hyperedge and len(hyperedge) >= 2:
                hyperedges.append(hyperedge)
                bridging_edges.append(hyperedge)
                dsu.union_many(chosen_roots)

        # Update metadata
        metadata.update(
            {
                "bridging_edges": bridging_edges,
                "s": s,
                "algorithm": "relaxed",
                "rho": rho,
                "load_cap": load_cap,
                "feasibility_matrix": F_comp,
            }
        )

        return cls(
            vertices=list(range(len(metadata["P_all"]))),
            hyperedges=hyperedges,
            metadata=metadata,
        )

    # ========================= Analysis Methods =========================

    def two_section(self) -> Set[Tuple[int, int]]:
        """
        Get the 2-section (primal graph) edges induced by hyperedges.

        Returns
        -------
        Set[Tuple[int, int]]
            Set of undirected edges (u, v) where u < v
        """
        if self._two_section_cache is not None:
            return self._two_section_cache

        edges = set()
        for he in self.hyperedges:
            if len(he) < 2:
                continue
            for u, v in it.combinations(sorted(he), 2):
                edges.add((u, v) if u < v else (v, u))

        self._two_section_cache = edges
        return edges

    def check_properties(self) -> Dict:
        """
        Verify hypergraph properties and theoretical bounds.

        Returns
        -------
        Dict
            Property checks including:
            - strict_ok: Each bridging hyperedge has ≤1 vertex per component
            - induced_ok: 2-section preserves component structure
            - connected_ok: 2-section is connected
            - bounds_ok: Number of bridging edges within theoretical bounds
            - optimal_ok: Achieves lower bound (for strict construction)
        """
        meta = self.metadata
        if not meta:
            return {"error": "No metadata available for property checking"}

        comp_of = meta.get("comp_of", [])
        bridging_edges = meta.get("bridging_edges", [])
        m = meta.get("m", 0)
        s = meta.get("s", 2)

        # Check strict bridging property
        strict_ok = self._check_strict_bridging(bridging_edges, comp_of, s)

        # Check induced subgraph property
        induced_ok, violations = self._check_induced_property()

        # Check connectivity
        connected_ok = self._check_connectivity()

        # Check theoretical bounds
        used_bridging = len(bridging_edges)
        lower_bound = math.ceil((m - 1) / (s - 1)) if m > 0 and s > 1 else 0
        upper_bound = m - 1 if m > 0 else 0
        bounds_ok = lower_bound <= used_bridging <= upper_bound
        optimal_ok = used_bridging == lower_bound

        return {
            "strict_ok": strict_ok,
            "induced_ok": induced_ok,
            "induced_violations": violations,
            "connected_ok": connected_ok,
            "bounds_ok": bounds_ok,
            "optimal_ok": optimal_ok,
            "used_bridging": used_bridging,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "m": m,
            "s": s,
        }

    # ========================= Visualization =========================

    def plot(
        self,
        title: str = "Hypergraph Visualization",
        save_path: Optional[str] = None,
        figsize: Tuple[float, float] = (8, 6),
        show_caption: bool = True,
    ) -> None:
        """
        Visualize the hypergraph with 2-section overlay.

        Parameters
        ----------
        title : str
            Figure title
        save_path : Optional[str]
            Path to save figure (if None, displays interactively)
        figsize : Tuple[float, float]
            Figure size in inches
        show_caption : bool
            Whether to show property summary caption
        """
        meta = self.metadata
        P = meta.get("P_all")
        if P is None:
            raise ValueError("No position data available for plotting")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot components with different colors
        comp_vertices = meta.get("comp_vertices", [])
        for cid, verts in enumerate(comp_vertices):
            if not verts:
                continue
            coords = P[verts]
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                label=f"Component {cid}",
                alpha=0.9,
                s=30,
            )

        # Plot intra-component edges
        intra_edges = meta.get("intra_edges_global", [])
        for u, v in intra_edges:
            ax.plot(
                [P[u, 0], P[v, 0]],
                [P[u, 1], P[v, 1]],
                "k-",
                linewidth=0.5,
                alpha=0.4,
            )

        # Plot bridging hyperedges
        bridging_edges = meta.get("bridging_edges", [])
        for he in bridging_edges:
            vertices = sorted(list(he))
            # Draw pairwise connections within hyperedge
            for u, v in it.combinations(vertices, 2):
                ax.plot(
                    [P[u, 0], P[v, 0]],
                    [P[u, 1], P[v, 1]],
                    "r--",
                    linewidth=2.0,
                    alpha=0.7,
                )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best", frameon=True, fontsize=9)
        ax.grid(True, alpha=0.2)

        # Add caption with properties
        if show_caption:
            checks = self.check_properties()
            m = checks.get("m", 0)
            s = checks.get("s", 0)
            used = checks.get("used_bridging", 0)
            lo = checks.get("lower_bound", 0)
            hi = checks.get("upper_bound", 0)
            opt = checks.get("optimal_ok", False)
            caption = (
                f"m={m}, s={s}, bridging={used} "
                f"(bounds [{lo}, {hi}], optimal={opt})"
            )
            fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=10)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    # ========================= PyG Integration =========================

    def to_pyg_hetero(self) -> HeteroData:
        """
        Convert to PyG HeteroData representing the incidence bipartite graph.

        Returns
        -------
        HeteroData
            Bipartite graph with node types "node" and "hyperedge"
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch Geometric not available")

        H = HeteroData()
        meta = self.metadata
        N = len(self.vertices)
        M = len(self.hyperedges)

        # Node features
        H["node"].num_nodes = N
        if "P_all" in meta:
            H["node"].pos = torch.tensor(meta["P_all"], dtype=torch.float32)
        if "comp_of" in meta:
            H["node"].comp = torch.tensor(meta["comp_of"], dtype=torch.long)

        # Hyperedge features
        H["hyperedge"].num_nodes = M
        he_order = [len(he) for he in self.hyperedges]
        H["hyperedge"].order = torch.tensor(he_order, dtype=torch.long)

        # Build incidence edges
        src_nodes, dst_hes = [], []
        for he_id, he in enumerate(self.hyperedges):
            for v in he:
                src_nodes.append(v)
                dst_hes.append(he_id)

        if src_nodes:
            inc_index = torch.tensor(
                np.vstack([src_nodes, dst_hes]), dtype=torch.long
            )
            H[("node", "in", "hyperedge")].edge_index = inc_index
            H[("hyperedge", "contains", "node")].edge_index = inc_index.flip(0)

        return H

    def to_pyg_data(self) -> Data:
        """
        Convert to PyG Data representing the 2-section graph.

        Returns
        -------
        Data
            Standard PyG graph with positions and component labels
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch Geometric not available")

        meta = self.metadata
        E2 = sorted(list(self.two_section()))

        if E2:
            ei = np.array(E2).T
            edge_index = torch.tensor(
                np.hstack([ei, ei[::-1, :]]), dtype=torch.long
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        data = Data(edge_index=edge_index)

        if "P_all" in meta:
            data.pos = torch.tensor(meta["P_all"], dtype=torch.float32)
        if "comp_of" in meta:
            data.comp = torch.tensor(meta["comp_of"], dtype=torch.long)

        return data

    # ========================= Private Helper Methods =========================

    @staticmethod
    def _prepare_metadata(
        components_pts: List[np.ndarray],
        edges_per_component: List[Iterable[Tuple[int, int]]],
    ) -> Dict:
        """Prepare metadata from component data."""
        # Concatenate all points
        comp_vertices = []
        comp_of = []
        current = 0
        all_pts = []

        for cid, pts in enumerate(components_pts):
            n = len(pts)
            ids = list(range(current, current + n))
            comp_vertices.append(ids)
            comp_of.extend([cid] * n)
            all_pts.append(pts)
            current += n

        P_all = np.vstack(all_pts) if all_pts else np.zeros((0, 2))

        # Process edges to global indices
        intra_edges_global = []
        for cid, local_edges in enumerate(edges_per_component):
            for u_local, v_local in local_edges:
                u = comp_vertices[cid][u_local]
                v = comp_vertices[cid][v_local]
                if u != v:
                    a, b = (u, v) if u < v else (v, u)
                    intra_edges_global.append((a, b))

        # Find representatives (closest to centroid)
        reps_global = []
        for cid, verts in enumerate(comp_vertices):
            if not verts:
                reps_global.append(-1)
            else:
                pts = P_all[verts]
                centroid = pts.mean(axis=0)
                d = np.linalg.norm(pts - centroid, axis=1)
                reps_global.append(verts[int(np.argmin(d))])

        return {
            "P_all": P_all,
            "comp_vertices": comp_vertices,
            "comp_of": comp_of,
            "intra_edges_global": intra_edges_global,
            "reps_global": reps_global,
            "m": len(components_pts),
        }

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
        P_all: np.ndarray,
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
        if rho is None:
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

    def _check_strict_bridging(
        self, bridging_edges: List[frozenset], comp_of: List[int], s: int
    ) -> bool:
        """Check if bridging satisfies strict constraints."""
        for he in bridging_edges:
            # Check size constraint
            if len(he) > s:
                return False

            # Check at most one vertex per component
            comps_in_he = [comp_of[v] for v in he]
            if len(set(comps_in_he)) < 2:  # Must span at least 2 components
                return False
            if len(comps_in_he) != len(set(comps_in_he)):  # Duplicates
                return False

        return True

    def _check_induced_property(self) -> Tuple[bool, List]:
        """Check if 2-section preserves component edge structure."""
        meta = self.metadata
        comp_vertices = meta.get("comp_vertices", [])
        intra_edges_set = set(meta.get("intra_edges_global", []))
        E2 = self.two_section()

        violations = []
        for cid, verts in enumerate(comp_vertices):
            verts_set = set(verts)
            # Edges in 2-section within this component
            intra_from_E2 = {
                (u, v) for (u, v) in E2 if u in verts_set and v in verts_set
            }
            # Original edges in this component
            supplied = {
                (u, v) for (u, v) in intra_edges_set if u in verts_set and v in verts_set
            }
            extra = intra_from_E2 - supplied
            if extra:
                violations.append((cid, extra))

        return len(violations) == 0, violations

    def _check_connectivity(self) -> bool:
        """Check if 2-section graph is connected."""
        N = len(self.vertices)
        if N == 0:
            return True

        # Build adjacency
        adj = {v: set() for v in self.vertices}
        for u, v in self.two_section():
            adj[u].add(v)
            adj[v].add(u)

        # BFS from vertex 0
        seen = set()
        stack = [0]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(adj[x] - seen)

        return len(seen) == N

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
            # Could add other strategies here
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

        # Could sort by distance or other criteria
        # For now, just take first k
        return nbrs[:k]

    @staticmethod
    def _form_hyperedge(
        chosen_roots: List[int],
        dsu,
        metadata: Dict,
        vertex_load: Dict,
        load_cap: Optional[int],
    ) -> Optional[frozenset[int]]:
        """Form a hyperedge from chosen roots."""
        comp_vertices = metadata["comp_vertices"]
        P_all = metadata["P_all"]
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

            if best_v is not None:
                used.append(best_v)
                seen_components.add(comp_of[best_v])
                vertex_load[best_v] = vertex_load.get(best_v, 0) + 1

        if len(used) >= 2:
            return frozenset(used)
        return None

    @staticmethod
    def _force_nearest_connection(dsu, metadata: Dict, vertex_load: Dict) -> Optional[frozenset[int]]:
        """Force connection between nearest pair of roots."""
        roots = dsu.all_roots()
        if len(roots) <= 1:
            return None

        # Find nearest pair of components
        P_all = metadata["P_all"]
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
                            # Find the actual closest vertices
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

        def union_many(self, roots: Iterable[int]) -> int:
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


def generate_demo_components(
    n_components: int = 5,
    n_per_component: int = 20,
    cluster_radius: float = 5.0,
    blob_std: float = 0.5,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[List[Tuple[int, int]]]]:
    """
    Generate demo component data with random 2D spatial distribution.

    Parameters
    ----------
    n_components : int
        Number of components to generate
    n_per_component : int
        Number of points per component
    cluster_radius : float
        Maximum radius for placing component centers
    blob_std : float
        Standard deviation for points within each blob
    seed : int
        Random seed for reproducibility

    Returns
    -------
    Tuple[List[np.ndarray], List[List[Tuple[int, int]]]]
        Component points and edge lists
    """
    rng = np.random.default_rng(seed)

    # Generate random centers in 2D space
    centers = []
    if n_components == 1:
        centers.append(np.array([0.0, 0.0]))
    else:
        # Place first center at origin
        centers.append(np.array([0.0, 0.0]))
        
        # Place remaining centers randomly but somewhat clustered
        for i in range(1, n_components):
            # Generate random angle and radius
            angle = rng.uniform(0, 2 * np.pi)
            # Use sqrt for uniform distribution in circle
            r = np.sqrt(rng.uniform(0.3, 1.0)) * cluster_radius
            center = np.array([r * np.cos(angle), r * np.sin(angle)])
            
            # Ensure minimum separation between centers
            while any(np.linalg.norm(center - c) < 1.5 for c in centers):
                angle = rng.uniform(0, 2 * np.pi)
                r = np.sqrt(rng.uniform(0.3, 1.0)) * cluster_radius
                center = np.array([r * np.cos(angle), r * np.sin(angle)])
            
            centers.append(center)

    components_pts = []
    edges_per_component = []

    for center in centers:
        # Generate blob points around center
        pts = center + blob_std * rng.standard_normal((n_per_component, 2))
        components_pts.append(pts)

        # Create edges - mix of chain and some random connections for variety
        edges = []
        # Chain backbone
        for j in range(n_per_component - 1):
            edges.append((j, j + 1))
        # Add a few random edges for more interesting structure
        n_extra = min(n_per_component // 4, 5)
        for _ in range(n_extra):
            a, b = rng.choice(n_per_component, 2, replace=False)
            if a > b:
                a, b = b, a
            if (a, b) not in edges:
                edges.append((a, b))
        
        edges_per_component.append(edges)

    return components_pts, edges_per_component


def demo():
    """Demonstrate both strict and relaxed hypergraph construction."""
    print("=" * 60)
    print("Hypergraph Construction Demo")
    print("=" * 60)

    # Generate demo data with random 2D distribution
    components, edges = generate_demo_components(
        n_components=5,
        n_per_component=15,
        cluster_radius=6.0,
        blob_std=0.4,
        seed=42
    )

    # Build strict hypergraph
    print("\n1. Building STRICT hypergraph (star batching)...")
    hg_strict = Hypergraph.build_strict(
        components_pts=components,
        edges_per_component=edges,
        s=3,
        hub_component=0
    )

    checks_strict = hg_strict.check_properties()
    print(f"   Bridging hyperedges used: {checks_strict['used_bridging']}")
    print(f"   Theoretical bounds: [{checks_strict['lower_bound']}, {checks_strict['upper_bound']}]")
    print(f"   Optimal: {checks_strict['optimal_ok']}")
    print(f"   All properties satisfied: {checks_strict['strict_ok'] and checks_strict['connected_ok']}")

    # Build relaxed hypergraph (fully connected)
    print("\n2. Building RELAXED hypergraph (no constraints)...")
    hg_relaxed_full = Hypergraph.build_relaxed(
        components_pts=components,
        edges_per_component=edges,
        s=3,
        rho=None
    )

    checks_relaxed = hg_relaxed_full.check_properties()
    print(f"   Bridging hyperedges used: {checks_relaxed['used_bridging']}")
    print(f"   Theoretical bounds: [{checks_relaxed['lower_bound']}, {checks_relaxed['upper_bound']}]")
    print(f"   Connected: {checks_relaxed['connected_ok']}")

    # Build relaxed hypergraph (with radius constraint)
    print("\n3. Building RELAXED hypergraph (radius constraint ρ=0.5)...")
    hg_relaxed_radius = Hypergraph.build_relaxed(
        components_pts=components,
        edges_per_component=edges,
        s=4,
        rho=0.5,
        force_connect=True
    )

    checks_radius = hg_relaxed_radius.check_properties()
    print(f"   Bridging hyperedges used: {checks_radius['used_bridging']}")
    print(f"   Connected: {checks_radius['connected_ok']}")

    # Visualize
    print("\n4. Generating visualizations...")
    hg_strict.plot(
        title="Strict Hypergraph (Star Batching)",
        save_path="hypergraph_strict.png",
        figsize=(10, 10)
    )
    hg_relaxed_full.plot(
        title="Relaxed Hypergraph (No Constraints)",
        save_path="hypergraph_relaxed_full.png",
        figsize=(10, 10)
    )
    hg_relaxed_radius.plot(
        title="Relaxed Hypergraph (ρ=0.5)",
        save_path="hypergraph_relaxed_radius.png",
        figsize=(10, 10)
    )
    print("   Saved: hypergraph_strict.png, hypergraph_relaxed_full.png, hypergraph_relaxed_radius.png")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()


# Export main components
__all__ = [
    "Hypergraph",
    "generate_demo_components",
    "demo",
    "quick_demo",
]