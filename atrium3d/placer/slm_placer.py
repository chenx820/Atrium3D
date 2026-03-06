from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

Point2 = Tuple[float, float]
Gate = Tuple[int, int]


def _ensure_compilation_time(results_code: Dict) -> Dict:
    ct = results_code.setdefault("compilation_time", {})
    ct.setdefault("total", 0.0)
    ct.setdefault("initial_mapping", 0.0)
    return results_code


def _euclidean(a: Point2, b: Point2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


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


def _mapping_cost(mapping: List[int], sites: Sequence[Point2], w: List[List[int]]) -> float:
    # mapping: qubit -> site_index
    n = len(mapping)
    cost = 0.0
    for i in range(n):
        si = sites[mapping[i]]
        wi = w[i]
        for j in range(i + 1, n):
            wij = wi[j]
            if wij:
                cost += wij * _euclidean(si, sites[mapping[j]])
    return cost


def _pick_central_sites(sites: Sequence[Point2], k: int) -> List[int]:
    # Rank by distance to centroid, pick k most central.
    cx = sum(p[0] for p in sites) / len(sites)
    cy = sum(p[1] for p in sites) / len(sites)
    ranked = sorted(range(len(sites)), key=lambda idx: _euclidean(sites[idx], (cx, cy)))
    return ranked[:k]


def _greedy_initial_mapping(
    sites: Sequence[Point2],
    w: List[List[int]],
) -> List[int]:
    n_qubits = len(w)
    central_site_ids = _pick_central_sites(sites, n_qubits)
    # Qubits ordered by total interaction weight (descending)
    qubit_order = sorted(range(n_qubits), key=lambda q: sum(w[q]), reverse=True)

    mapping = [-1] * n_qubits
    used_sites = set()

    # Seed: put the most interactive qubit at the most central site.
    mapping[qubit_order[0]] = central_site_ids[0]
    used_sites.add(central_site_ids[0])

    # Greedily place remaining qubits.
    for q in qubit_order[1:]:
        best_s = None
        best_score = float("inf")
        for s in central_site_ids:
            if s in used_sites:
                continue
            # Incremental cost: distance to already placed qubits weighted by interactions.
            score = 0.0
            for q2 in range(n_qubits):
                s2 = mapping[q2]
                if s2 == -1:
                    continue
                wij = w[q][q2]
                if wij:
                    score += wij * _euclidean(sites[s], sites[s2])
            if score < best_score:
                best_score = score
                best_s = s
        if best_s is None:
            # Fallback: any unused site
            for s in range(len(sites)):
                if s not in used_sites:
                    best_s = s
                    break
        mapping[q] = int(best_s)
        used_sites.add(int(best_s))

    return mapping


def _local_improve_swaps(
    mapping: List[int],
    sites: Sequence[Point2],
    w: List[List[int]],
    max_iters: int = 2000,
    seed: int = 0,
) -> List[int]:
    rng = random.Random(seed)
    best = mapping[:]
    best_cost = _mapping_cost(best, sites, w)
    n = len(best)

    # Random swap hill-climb.
    for _ in range(max_iters):
        a = rng.randrange(n)
        b = rng.randrange(n)
        if a == b:
            continue
        cand = best[:]
        cand[a], cand[b] = cand[b], cand[a]
        c = _mapping_cost(cand, sites, w)
        if c < best_cost:
            best, best_cost = cand, c
    return best


def placing_slm(
    qubit_slm_sites: Sequence[Point2],
    buffer_slm_sites: Sequence[Point2],
    results_code: Dict,
    list_full_gates: Sequence[Sequence[Gate]],
    qubits_mapping: Optional[Dict[int, Point2] | Sequence[Point2] | Sequence[int]] = None,
) -> Tuple[Dict, Dict[int, Point2]]:
    """
    2D SLM trap 上的初始映射（保留以兼容之前的实现）。
    """
    _ensure_compilation_time(results_code)
    t0 = time.perf_counter()

    n_qubits = int(results_code.get("n_qubits", 0))
    if n_qubits <= 0:
        raise ValueError("[Error] results_code['n_qubits'] must be set before placing_slm().")

    sites = list(qubit_slm_sites)
    if len(sites) < n_qubits:
        raise ValueError(f"[Error] #qubits ({n_qubits}) > #qubit_slm_sites ({len(sites)}).")

    if qubits_mapping is not None:
        mapping_dict: Dict[int, Point2] = {}
        if isinstance(qubits_mapping, dict):
            for q, p in qubits_mapping.items():
                mapping_dict[int(q)] = (float(p[0]), float(p[1]))
        else:
            if len(qubits_mapping) != n_qubits:
                raise ValueError("[Error] Given qubits_mapping length must equal n_qubits.")
            first = qubits_mapping[0]
            if isinstance(first, (tuple, list)) and len(first) == 2:
                for q in range(n_qubits):
                    p = qubits_mapping[q]
                    mapping_dict[q] = (float(p[0]), float(p[1]))
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
        return results_code, mapping_dict

    w = _interaction_weights(list_full_gates, n_qubits)
    base = _greedy_initial_mapping(sites, w)
    improved = _local_improve_swaps(base, sites, w, max_iters=2000, seed=0)

    best_mapping = {q: sites[improved[q]] for q in range(n_qubits)}
    results_code["initial_mapping"] = best_mapping

    elapsed = time.perf_counter() - t0
    results_code["compilation_time"]["initial_mapping"] = float(elapsed)
    results_code["compilation_time"]["total"] = float(results_code["compilation_time"]["total"] + elapsed)

    return results_code, best_mapping

