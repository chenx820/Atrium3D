from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

Point3 = Tuple[float, float, float]
Gate = Tuple[int, int]


def _ensure_compilation_time(results_code: Dict) -> Dict:
    ct = results_code.setdefault("compilation_time", {})
    ct.setdefault("total", 0.0)
    ct.setdefault("initial_mapping", 0.0)
    return results_code


def _euclidean3(a: Point3, b: Point3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _interaction_weights(list_full_gates: Sequence[Sequence[Gate]], n_qubits: int) -> List[List[int]]:
    w = [[0 for _ in range(n_qubits)] for _ in range(n_qubits)]
    for stage in list_full_gates:
        for q0, q1 in stage:
            if q0 == q1:
                continue
            if 0 <= q0 < n_qubits and 0 <= q1 < n_qubits:
                w[q0][q1] += 1
                w[q1][q0] += 1
    return w


def _mapping_cost(
    mapping: List[int],
    sites: Sequence[Point3],
    w: List[List[int]],
    readout_plane_z: Optional[float] = None,
    readout_urgency: Optional[Sequence[float]] = None,
    readout_weight: float = 0.0,
) -> float:
    n = len(mapping)
    cost = 0.0
    for i in range(n):
        si = sites[mapping[i]]
        if readout_weight and readout_plane_z is not None and readout_urgency is not None:
            # Earlier-finished qubits (higher urgency) should be closer (in z) to readout plane.
            # We minimize vertical travel: (readout_plane_z - z_site) for storage layers.
            dz = max(0.0, float(readout_plane_z) - float(si[2]))
            cost += float(readout_weight) * float(readout_urgency[i]) * dz
        wi = w[i]
        for j in range(i + 1, n):
            wij = wi[j]
            if wij:
                cost += wij * _euclidean3(si, sites[mapping[j]])
    return cost


def _sort_sites_by_centrality(sites: Sequence[Point3]) -> List[int]:
    cx = sum(p[0] for p in sites) / len(sites)
    cy = sum(p[1] for p in sites) / len(sites)
    cz = sum(p[2] for p in sites) / len(sites)
    c = (cx, cy, cz)
    return sorted(range(len(sites)), key=lambda i: _euclidean3(sites[i], c))


def _greedy_mapping_3d(
    sites: Sequence[Point3],
    w: List[List[int]],
    prefer_prefix: Optional[Sequence[int]] = None,
    readout_plane_z: Optional[float] = None,
    readout_urgency: Optional[Sequence[float]] = None,
    readout_weight: float = 0.0,
) -> List[int]:
    """
    qubit -> site_index 的贪心构造。
    prefer_prefix: 一个 site_index 列表，表示更优先被选择的候选集合（比如 interaction zone）。
    """
    n_qubits = len(w)
    all_ranked = _sort_sites_by_centrality(sites)

    if prefer_prefix:
        prefer_set = set(prefer_prefix)
        ranked = [i for i in all_ranked if i in prefer_set] + [i for i in all_ranked if i not in prefer_set]
    else:
        ranked = all_ranked

    qubit_order = sorted(range(n_qubits), key=lambda q: sum(w[q]), reverse=True)

    mapping = [-1] * n_qubits
    used = set()

    # seed
    mapping[qubit_order[0]] = ranked[0]
    used.add(ranked[0])

    for q in qubit_order[1:]:
        best_s = None
        best_score = float("inf")
        for s in ranked:
            if s in used:
                continue
            score = 0.0
            if readout_weight and readout_plane_z is not None and readout_urgency is not None:
                dz = max(0.0, float(readout_plane_z) - float(sites[s][2]))
                score += float(readout_weight) * float(readout_urgency[q]) * dz
            for q2 in range(n_qubits):
                s2 = mapping[q2]
                if s2 == -1:
                    continue
                wij = w[q][q2]
                if wij:
                    score += wij * _euclidean3(sites[s], sites[s2])
            if score < best_score:
                best_score = score
                best_s = s
        if best_s is None:
            raise RuntimeError("[Error] Not enough sites to place all qubits.")
        mapping[q] = int(best_s)
        used.add(int(best_s))

    return mapping


def _local_improve_swaps(
    mapping: List[int],
    sites: Sequence[Point3],
    w: List[List[int]],
    max_iters: int = 4000,
    seed: int = 0,
    readout_plane_z: Optional[float] = None,
    readout_urgency: Optional[Sequence[float]] = None,
    readout_weight: float = 0.0,
) -> List[int]:
    rng = random.Random(seed)
    best = mapping[:]
    best_cost = _mapping_cost(
        best,
        sites,
        w,
        readout_plane_z=readout_plane_z,
        readout_urgency=readout_urgency,
        readout_weight=readout_weight,
    )
    n = len(best)

    for _ in range(max_iters):
        a = rng.randrange(n)
        b = rng.randrange(n)
        if a == b:
            continue
        cand = best[:]
        cand[a], cand[b] = cand[b], cand[a]
        c = _mapping_cost(
            cand,
            sites,
            w,
            readout_plane_z=readout_plane_z,
            readout_urgency=readout_urgency,
            readout_weight=readout_weight,
        )
        if c < best_cost:
            best, best_cost = cand, c
    return best


def placing_3d(
    available_sites: Sequence[Point3],
    results_code: Dict,
    list_full_gates: Sequence[Sequence[Gate]],
    qubits_mapping: Optional[Dict[int, Point3] | Sequence[Point3] | Sequence[int]] = None,
    preferred_sites: Optional[Sequence[Point3]] = None,
    readout_plane_z: Optional[float] = None,
    readout_urgency: Optional[Sequence[float]] = None,
    readout_weight: float = 0.0,
) -> Tuple[Dict, Dict[int, Point3]]:
    """
    在 3D Atrium 架构上做初始放置：
    - available_sites: 可放置的 3D 物理坐标 (x,y,z)
    - preferred_sites: 可选的“优先区域”（比如 interaction zone 的 sites），用于引导贪心选点
    """
    _ensure_compilation_time(results_code)
    t0 = time.perf_counter()

    n_qubits = int(results_code.get("n_qubits", 0))
    if n_qubits <= 0:
        raise ValueError("[Error] results_code['n_qubits'] must be set before placing_3d().")

    sites = list(available_sites)
    if len(sites) < n_qubits:
        raise ValueError(f"[Error] #qubits ({n_qubits}) > #available_sites ({len(sites)}).")

    # normalize user mapping if provided
    if qubits_mapping is not None:
        mapping_dict: Dict[int, Point3] = {}
        if isinstance(qubits_mapping, dict):
            for q, p in qubits_mapping.items():
                mapping_dict[int(q)] = (float(p[0]), float(p[1]), float(p[2]))
        else:
            if len(qubits_mapping) != n_qubits:
                raise ValueError("[Error] Given qubits_mapping length must equal n_qubits.")
            first = qubits_mapping[0]
            if isinstance(first, (tuple, list)) and len(first) == 3:
                for q in range(n_qubits):
                    p = qubits_mapping[q]
                    mapping_dict[q] = (float(p[0]), float(p[1]), float(p[2]))
            else:
                used = set()
                for q in range(n_qubits):
                    idx = int(qubits_mapping[q])  # type: ignore[arg-type]
                    if idx < 0 or idx >= len(sites):
                        raise ValueError("[Error] Given qubits_mapping has invalid site index.")
                    if idx in used:
                        raise ValueError("[Error] Given qubits_mapping uses duplicate site index.")
                    used.add(idx)
                    mapping_dict[q] = sites[idx]

        results_code["initial_mapping"] = mapping_dict
        elapsed = time.perf_counter() - t0
        results_code["compilation_time"]["initial_mapping"] = float(elapsed)
        results_code["compilation_time"]["total"] = float(results_code["compilation_time"]["total"] + elapsed)
        results_code["initial_mapping_space"] = "3d"
        return results_code, mapping_dict

    w = _interaction_weights(list_full_gates, n_qubits)

    prefer_indices = None
    if preferred_sites:
        pref_set = set((float(x), float(y), float(z)) for (x, y, z) in preferred_sites)
        prefer_indices = [i for i, p in enumerate(sites) if (float(p[0]), float(p[1]), float(p[2])) in pref_set]

    base = _greedy_mapping_3d(
        sites,
        w,
        prefer_prefix=prefer_indices,
        readout_plane_z=readout_plane_z,
        readout_urgency=readout_urgency,
        readout_weight=readout_weight,
    )
    improved = _local_improve_swaps(
        base,
        sites,
        w,
        max_iters=4000,
        seed=0,
        readout_plane_z=readout_plane_z,
        readout_urgency=readout_urgency,
        readout_weight=readout_weight,
    )

    best_mapping = {q: sites[improved[q]] for q in range(n_qubits)}
    results_code["initial_mapping"] = best_mapping
    results_code["initial_mapping_space"] = "3d"

    elapsed = time.perf_counter() - t0
    results_code["compilation_time"]["initial_mapping"] = float(elapsed)
    results_code["compilation_time"]["total"] = float(results_code["compilation_time"]["total"] + elapsed)

    return results_code, best_mapping

