"""Structural analyses about the graph.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Set

from .io_utils import Dataset


def pose_adjacency(ds: Dataset, include_loops: bool = True) -> Dict[int, Set[int]]:
    adj = {i: set() for i in range(ds.n_poses)}
    for r in ds.odometry.itertuples(index=False):
        i, j = int(r.from_id), int(r.to_id)
        adj[i].add(j)
        adj[j].add(i)
    if include_loops:
        for r in ds.loops.itertuples(index=False):
            i, j = int(r.from_id), int(r.to_id)
            adj[i].add(j)
            adj[j].add(i)
    return adj


def markov_blanket(ds: Dataset, pose_id: int) -> dict:
    neighbors: Set[int] = set()
    odom, loop, gps, lm = [], [], [], []
    for r in ds.odometry.itertuples(index=False):
        i, j = int(r.from_id), int(r.to_id)
        if pose_id in (i, j):
            neighbors.add(j if pose_id == i else i)
            odom.append(f"odom({i},{j})")
    for r in ds.loops.itertuples(index=False):
        i, j = int(r.from_id), int(r.to_id)
        if pose_id in (i, j):
            neighbors.add(j if pose_id == i else i)
            loop.append(f"loop_{int(r.closure_id)}({i},{j})")
    for r in ds.gps.itertuples(index=False):
        if int(r.pose_id) == pose_id:
            gps.append(f"gps_{int(r.meas_id)}")
    for r in ds.landmark_observations.itertuples(index=False):
        if int(r.pose_id) == pose_id:
            lm.append(f"lm_{int(r.obs_id)}")
    return {
        "pose_id": pose_id,
        "neighbor_pose_ids": sorted(neighbors),
        "odometry_factors": odom,
        "loop_factors": loop,
        "gps_factors": gps,
        "landmark_factors": lm,
        "has_prior": pose_id == 0,
    }


def symbolic_fill_in(adjacency: Dict[int, Set[int]], ordering: List[int]) -> dict:
    g = deepcopy(adjacency)
    fill_edges = 0
    max_clique = 0
    for node in ordering:
        nbrs = list(g[node])
        max_clique = max(max_clique, len(nbrs) + 1)
        for a_i, a in enumerate(nbrs):
            for b in nbrs[a_i + 1:]:
                if b not in g[a]:
                    g[a].add(b)
                    g[b].add(a)
                    fill_edges += 1
        for nb in nbrs:
            g[nb].discard(node)
        g[node].clear()
    return {"fill_edges": fill_edges, "max_induced_clique_size": max_clique}


def greedy_min_degree_order(adjacency: Dict[int, Set[int]]) -> List[int]:
    g = deepcopy(adjacency)
    remaining = set(g)
    order: List[int] = []
    while remaining:
        node = min(remaining, key=lambda n: (len(g[n] & remaining), n))
        nbrs = list(g[node] & remaining)
        for a_i, a in enumerate(nbrs):
            for b in nbrs[a_i + 1:]:
                g[a].add(b)
                g[b].add(a)
        remaining.remove(node)
        order.append(node)
    return order
