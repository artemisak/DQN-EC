# Hypergraph builder & demo per your spec
# - Input: coordinate pairs per component and edges per component
# - Output: hypergraph object, property/bounds checks in console, and a figure
#
# Notes:
# • Uses Matplotlib (no seaborn, single plot, no explicit colors set).
# • "Strict bridging": at most one vertex from any component in a bridging hyperedge.
# • Batching parameter s controls the max size of a bridging hyperedge.
#
# You can reuse the functions below with your own coordinates & edges. 
# A demo with 3 random 2-D kNN graphs is included at the end.

from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Iterable
import numpy as np
import itertools as it
import math
import matplotlib.pyplot as plt

# ----------------------------- core data structures -----------------------------

@dataclass(frozen=True)
class Hypergraph:
    V: List[int]                       # global vertex ids [0..N-1]
    E: List[frozenset]                 # list of hyperedges (each a frozenset of global ids)


# ----------------------------- helpers -----------------------------

def pairwise_distances(X: np.ndarray) -> np.ndarray:
    # Euclidean pairwise distances (n x n); X shape = (n,2)
    diffs = X[:, None, :] - X[None, :, :]
    return np.sqrt((diffs ** 2).sum(axis=2))


def symmetric_knn_edges(points: np.ndarray, k: int) -> Set[Tuple[int, int]]:
    """
    Undirected symmetric kNN in 2D.
    Returns local index pairs (i<j) such that i in kNN(j) or j in kNN(i).
    """
    n = len(points)
    if n <= 1 or k <= 0:
        return set()
    D = pairwise_distances(points)
    np.fill_diagonal(D, np.inf)
    # for each j, get indices of k nearest neighbors
    k = min(k, n - 1)
    nbrs = np.argpartition(D, kth=k-1, axis=1)[:, :k]
    E = set()
    for j in range(n):
        for i in nbrs[j]:
            a, b = sorted((i, j))
            E.add((a, b))
    return E


def concat_components(components_pts: List[np.ndarray]) -> Tuple[np.ndarray, List[List[int]], List[int]]:
    """
    Concatenate component point arrays into a single array.
    Returns:
      P_all: (N,2) array
      comp_vertices: list of lists mapping component -> list of global ids
      comp_of: list of component id for each global vertex
    """
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
    return P_all, comp_vertices, comp_of


def to_global_edges(local_edges: Iterable[Tuple[int, int]], comp_ids: List[int]) -> Set[Tuple[int, int]]:
    """(Not used—kept for completeness)."""
    return set(local_edges)


def component_representative(pts: np.ndarray) -> int:
    """
    Choose a representative (index) for a component: nearest to its centroid.
    """
    if len(pts) == 0:
        return -1
    centroid = pts.mean(axis=0)
    d = np.linalg.norm(pts - centroid, axis=1)
    return int(np.argmin(d))


# ----------------------------- construction logic -----------------------------

def build_hypergraph_strict(
    components_pts: List[np.ndarray],
    edges_per_component_local: List[Iterable[Tuple[int, int]]],
    s: int,
    hub_component: int = 0,
) -> Tuple[Hypergraph, Dict]:
    """
    Build a strict-bridging hypergraph:
      - Add all intra-component edges as 2-hyperedges
      - Add bridging hyperedges of size <= s with at most one vertex per component,
        using the optimal star batching construction.
    Returns the hypergraph and a dict of metadata.
    """
    assert len(components_pts) == len(edges_per_component_local)
    m = len(components_pts)
    P_all, comp_vertices, comp_of = concat_components(components_pts)
    V = list(range(len(P_all)))
    H_edges = []

    # Map (component, local_idx) -> global id
    def g(c, i_local): 
        return comp_vertices[c][i_local]

    # 1) add intra-component edges as 2-hyperedges
    intra_edges_global = set()
    for c, local_edges in enumerate(edges_per_component_local):
        for (u_local, v_local) in local_edges:
            u, v = g(c, u_local), g(c, v_local)
            a, b = (u, v) if u < v else (v, u)
            intra_edges_global.add((a, b))
            H_edges.append(frozenset({a, b}))

    # 2) representatives (one per component)
    reps_local = [component_representative(components_pts[c]) for c in range(m)]
    reps_global = [g(c, reps_local[c]) for c in range(m)]

    # 3) optimal star batching for bridging hyperedges
    others = [j for j in range(m) if j != hub_component]
    block_size = max(1, s - 1)  # each bridging hyperedge includes hub rep + up to s-1 others
    bridging_edges = []
    for start in range(0, len(others), block_size):
        block = others[start : start + block_size]
        he = {reps_global[hub_component]} | {reps_global[j] for j in block}
        if len(he) >= 2:
            H_edges.append(frozenset(he))
            bridging_edges.append(frozenset(he))

    H = Hypergraph(V=V, E=H_edges)

    metadata = dict(
        P_all=P_all,
        comp_vertices=comp_vertices,
        comp_of=comp_of,
        intra_edges_global=intra_edges_global,
        reps_global=reps_global,
        bridging_edges=bridging_edges,
        m=m,
        s=s,
        hub_component=hub_component,
    )
    return H, metadata


# ----------------------------- derived graphs & checks -----------------------------

def two_section_edges(H: Hypergraph) -> Set[Tuple[int, int]]:
    """
    All pairwise edges induced by hyperedges (the 2-section / primal graph).
    """
    edges = set()
    for e in H.E:
        if len(e) < 2:
            continue
        for u, v in it.combinations(sorted(e), 2):
            edges.add((u, v))
    return edges


def check_induced_subgraphs(meta: Dict) -> Tuple[bool, List[Tuple[int, Set[Tuple[int,int]]]]]:
    """
    For each component, assert it remains an induced subgraph of the 2-section:
      2-section should not create extra edges inside the component beyond the given kNN edges.
    Returns (ok, details). 'details' lists per-component extraneous edges (if any).
    """
    P_all = meta["P_all"]
    comp_vertices = meta["comp_vertices"]
    intra_edges_global = meta["intra_edges_global"]

    # Build set of all 2-section edges
    # (Requires H; but we can reconstruct H from meta fields in this scope only if we pass H)
    # We'll compute outside and pass into this function normally; for convenience, patch at call site.

    raise NotImplementedError("Use check_all_properties(...) which bundles everything.")


def check_all_properties(H: Hypergraph, meta: Dict) -> Dict:
    """
    Check:
      - Strict bridging: each bridging hyperedge has at most one vertex per component
      - Induced: no extra intra-component edges in 2-section beyond original
      - Global connectivity (2-section is connected)
      - Bounds on #bridging hyperedges: ceil((m-1)/(s-1)) ≤ used ≤ m-1 and equality to lower bound
    Returns a dict with booleans and details.
    """
    comp_of = meta["comp_of"]
    comp_vertices = meta["comp_vertices"]
    intra_edges_global = meta["intra_edges_global"]
    bridging_edges = meta["bridging_edges"]
    m, s = meta["m"], meta["s"]

    # 2-section edges
    E2 = two_section_edges(H)

    # Strict bridging check
    strict_ok = True
    for e in bridging_edges:
        comps_in_e = [comp_of[v] for v in e]
        # size must be >= 2 and include at least two distinct components
        if len(set(comps_in_e)) < 2:
            strict_ok = False
            break
        # at most one vertex from any component
        counts = {}
        for c in comps_in_e:
            counts[c] = counts.get(c, 0) + 1
            if counts[c] > 1:
                strict_ok = False
                break
        if not strict_ok:
            break
        # size cap
        if len(e) > s:
            strict_ok = False
            break

    # Induced-subgraph check inside each component
    induced_ok = True
    induced_violations = []
    intra_edges_set = set(intra_edges_global)
    for cid, verts in enumerate(comp_vertices):
        verts_set = set(verts)
        # edges in 2-section induced within component
        intra_from_E2 = {(u, v) for (u, v) in E2 if u in verts_set and v in verts_set}
        # must equal the supplied intra edges restricted to this component
        supplied_intra_c = {(u, v) for (u, v) in intra_edges_set if u in verts_set and v in verts_set}
        extra = intra_from_E2 - supplied_intra_c
        if extra:
            induced_ok = False
            induced_violations.append((cid, extra))

    # Connectivity check on 2-section
    N = len(H.V)
    adj = {v: set() for v in H.V}
    for u, v in E2:
        adj[u].add(v)
        adj[v].add(u)
    # BFS from an arbitrary vertex (if any vertices exist)
    connected_ok = True
    if N > 0:
        seen = set()
        stack = [0]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(adj[x] - seen)
        connected_ok = (len(seen) == N)

    # Bounds check
    used_bridging = len(bridging_edges)
    lower = math.ceil((m - 1) / max(1, (s - 1)))
    upper = m - 1
    bounds_ok = (lower <= used_bridging <= upper)
    optimal_ok = (used_bridging == lower)

    return dict(
        strict_ok=strict_ok,
        induced_ok=induced_ok,
        induced_violations=induced_violations,
        connected_ok=connected_ok,
        bounds_ok=bounds_ok,
        optimal_ok=optimal_ok,
        used_bridging=used_bridging,
        lower_bound=lower,
        upper_bound=upper,
        m=m,
        s=s,
    )


# ----------------------------- plotting -----------------------------

def plot_hypergraph(H: Hypergraph, meta: Dict, title: str = "Hypergraph (2-section overlay)"):
    """
    Single-axes Matplotlib plot:
      • Scatter all points, grouped by component (default color cycle).
      • Draw intra-component edges as thin lines.
      • Draw bridging hyperedges by connecting all pairs in each bridging edge (thicker, dashed).
    """
    P = meta["P_all"]
    comp_vertices = meta["comp_vertices"]
    intra_edges_global = meta["intra_edges_global"]
    bridging_edges = meta["bridging_edges"]

    fig = plt.figure(figsize=(7, 6))
    ax = plt.gca()

    # scatter per component
    for cid, verts in enumerate(comp_vertices):
        coords = P[verts]
        ax.scatter(coords[:, 0], coords[:, 1], label=f"Comp {cid}", alpha=0.9)

    # intra-component edges
    for (u, v) in intra_edges_global:
        x = [P[u, 0], P[v, 0]]
        y = [P[u, 1], P[v, 1]]
        ax.plot(x, y, linewidth=0.6, alpha=0.6)

    # bridging hyperedges
    for e in bridging_edges:
        e_list = sorted(list(e))
        # connect all pairs in the hyperedge
        for u, v in it.combinations(e_list, 2):
            x = [P[u, 0], P[v, 0]]
            y = [P[u, 1], P[v, 1]]
            ax.plot(x, y, linewidth=2.0, linestyle="--", alpha=0.9)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    plt.show()


# ----------------------------- user-facing wrapper -----------------------------

def build_and_plot_hypergraph_from_components(
    components_pts: List[np.ndarray],
    edges_per_component_local: List[Iterable[Tuple[int, int]]],
    s: int,
    hub_component: int = 0,
    plot_title: str = "Hypergraph (2-section overlay)",
):
    """
    Convenience wrapper: builds the strict-bridging hypergraph and plots it,
    then prints the property/bounds checks.
    """
    H, meta = build_hypergraph_strict(components_pts, edges_per_component_local, s, hub_component=hub_component)
    checks = check_all_properties(H, meta)

    # Console report
    print("=== Hypergraph construction report ===")
    print(f"Components (m):        {checks['m']}")
    print(f"Max hyperedge size s:  {checks['s']}")
    print(f"Bridging hyperedges:   {checks['used_bridging']}")
    print(f"Lower bound:           {checks['lower_bound']}  (ceil((m-1)/(s-1)))")
    print(f"Upper bound:           {checks['upper_bound']}  (m-1)")
    print(f"Bounds satisfied?      {checks['bounds_ok']}")
    print(f"Optimal (=lower)?      {checks['optimal_ok']}")
    print(f"Strict bridging OK?    {checks['strict_ok']}")
    print(f"Induced subgraphs OK?  {checks['induced_ok']}")
    if not checks["induced_ok"]:
        print("  Violations per component (extra intra edges detected):")
        for cid, extra in checks["induced_violations"]:
            print(f"   - Component {cid}: {sorted(list(extra))}")
    print(f"2-section connected?   {checks['connected_ok']}")

    plot_hypergraph(H, meta, title=plot_title)

    return H, meta, checks


# ----------------------------- demo: 3 random 2-D kNN graphs near each other -----------------------------

def demo_three_random_knn_graphs(n_per_comp=(20, 20, 20), k_per_comp=(4, 4, 4), s=3, seed=7):
    """
    Create three close-by Gaussian clusters in 2D, build symmetric kNN graphs on each,
    then batch-connect them with strict bridging hyperedges of size ≤ s.
    """
    rng = np.random.default_rng(seed)

    # Cluster centers positioned close to each other
    centers = np.array([
        [0.0, 0.0],
        [2.2, 0.3],
        [1.2, 2.3],
    ])

    components_pts = []
    edges_per_component_local = []

    for idx, (n, k) in enumerate(zip(n_per_comp, k_per_comp)):
        pts = centers[idx] + 0.4 * rng.standard_normal(size=(n, 2))
        components_pts.append(pts)
        E_local = symmetric_knn_edges(pts, k=k)
        edges_per_component_local.append(sorted(E_local))

    print("Built symmetric 2-D kNN graphs per component.")
    print("Edge counts per component:", [len(e) for e in edges_per_component_local])

    H, meta, checks = build_and_plot_hypergraph_from_components(
        components_pts,
        edges_per_component_local,
        s=s,
        hub_component=0,
        plot_title="Demo: 3 close kNN graphs with strict bridging",
    )
    return H, meta, checks


# ----------------------------- run demo -----------------------------

# You can comment this out and call demo_three_random_knn_graphs(...) manually with your own parameters.
H_demo, meta_demo, checks_demo = demo_three_random_knn_graphs()
