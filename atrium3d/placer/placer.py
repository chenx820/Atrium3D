from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

Point3 = Tuple[float, float, float]
Gate = Tuple[int, int]


def _euclidean3(a: Point3, b: Point3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _grid_to_phys(site: Tuple[int, int, int], spacing_xy: float, spacing_z: float) -> Point3:
    x, y, z = site
    return (float(x * spacing_xy), float(y * spacing_xy), float(z * spacing_z))


def _phys_to_xy_col(p: Point3, spacing_xy: float) -> Tuple[int, int]:
    # Positions are always generated as multiples of spacing; round for safety.
    return (int(round(p[0] / spacing_xy)), int(round(p[1] / spacing_xy)))


def _interaction_sites_grid(size: int, layers: int, center_range: range) -> List[Tuple[int, int, int]]:
    sites = []
    for z in range(layers):
        if z == layers - 1:
            continue
        for x in center_range:
            for y in center_range:
                sites.append((x, y, z))
    return sites


def _interaction_adjacent_pairs(
    layers: int,
    center_range: range,
) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """
    生成可做两比特门的 interaction 区“相邻对”。
    当前采用同一层、XY 曼哈顿距离=1 的相邻对。
    """
    pairs: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []
    xs = list(center_range)
    ys = list(center_range)
    for z in range(layers - 1):
        for x in xs:
            for y in ys:
                if x + 1 in center_range:
                    pairs.append(((x, y, z), (x + 1, y, z)))
                if y + 1 in center_range:
                    pairs.append(((x, y, z), (x, y + 1, z)))
    return pairs


def place_stages(
    *,
    initial_mapping: Dict[int, Point3],
    stages: Sequence[Sequence[Gate]],
    size: int,
    layers: int,
    center_range: range,
    spacing_xy: float,
    spacing_z: float,
    last_stage_2q: Optional[Sequence[int]] = None,
    enable_readout_move: bool = True,
) -> Tuple[List[Dict[int, Point3]], List[Dict], Dict]:
    """
    为每个 stage 决定原子（qubit）的放置位置。

    设计目标（当前最小可用版本）：
    - stage 内每个两比特门 (q0,q1) 把两原子放到 interaction 区相邻位，便于执行门
    - stage 内避免 skewer 串扰：同一 stage 所有被选中的 (x,y) 列必须互不相同
    - 代价函数：最小化从当前位到目标位的搬运距离（逐 stage 更新当前位置）
    - 可选：当 qubit 完成最后一次两比特门后，stage 结束把它上移到 readout plane（同一 x,y 列）

    返回：
    - stage_positions: list[dict[qubit->(x,y,z)]]，表示每个 stage 结束后的 qubit 位置
    - stage_meta: list[dict]，每个 stage 的 gate->site 细节与搬运距离
    - summary: dict，总搬运距离等摘要
    """
    n_qubits = len(initial_mapping)
    current_pos: Dict[int, Point3] = {int(q): (float(p[0]), float(p[1]), float(p[2])) for q, p in initial_mapping.items()}

    # Candidate interaction site pairs for 2Q gates
    pair_grid = _interaction_adjacent_pairs(layers=layers, center_range=center_range)
    pair_phys = [(_grid_to_phys(a, spacing_xy, spacing_z), _grid_to_phys(b, spacing_xy, spacing_z), a, b) for (a, b) in pair_grid]

    readout_plane_z = float((layers - 1) * spacing_z)

    # NOTE: scheduler 的 stage 只保证“qubit 不冲突”，但不保证 interaction 区资源足够。
    # 这里会把每个 scheduler stage 进一步切分为多个 micro-stage（顺序执行），以满足：
    # - interaction 的 (x,y) 列容量（skewer-aware）
    # - 每个 2Q gate 需要占用两个不同列
    stage_positions: List[Dict[int, Point3]] = []  # micro-stage end positions
    stage_meta: List[Dict] = []  # micro-stage meta (includes original stage index)

    total_travel = 0.0

    interaction_cols_capacity = int(len(center_range) * len(center_range))  # number of (x,y) columns in interaction zone
    max_gates_per_micro = max(1, interaction_cols_capacity // 2)

    micro_stage_idx = 0

    for orig_stage_idx, stage in enumerate(stages):
        # Only 2Q gates need interaction placement. Scheduler guarantees qubit-disjoint within stage.
        remaining: List[Gate] = [(int(q0), int(q1)) for (q0, q1) in stage if int(q0) != int(q1)]

        # Helper: difficulty ordering
        def _gate_difficulty(g: Gate) -> float:
            q0, q1 = g
            p0 = current_pos[q0]
            p1 = current_pos[q1]
            cx = float((center_range.start + center_range.stop - 1) * 0.5 * spacing_xy)
            cy = float((center_range.start + center_range.stop - 1) * 0.5 * spacing_xy)
            cz = float((layers - 2) * spacing_z)
            return _euclidean3(p0, (cx, cy, cz)) + _euclidean3(p1, (cx, cy, cz))

        # If stage has no 2Q gates, still advance a micro-stage snapshot (positions unchanged)
        if not remaining:
            stage_positions.append({int(q): current_pos[q] for q in range(n_qubits)})
            stage_meta.append(
                {
                    "micro_stage": int(micro_stage_idx),
                    "original_stage": int(orig_stage_idx),
                    "micro_in_original": 0,
                    "two_qubit_gates": [],
                    "gate_sites": [],
                    "readout_moves": [],
                    "travel_distance": 0.0,
                }
            )
            micro_stage_idx += 1
            continue

        micro_in_original = 0

        while remaining:
            # Try to pack up to capacity gates into this micro-stage.
            remaining.sort(key=_gate_difficulty, reverse=True)
            candidate_order = remaining[:]

            used_xy_cols = set()
            used_sites = set()
            placements: Dict[int, Point3] = {}
            gate_sites: List[Dict] = []
            stage_travel = 0.0

            assigned: List[Gate] = []
            skipped: List[Gate] = []

            for (q0, q1) in candidate_order:
                if len(assigned) >= max_gates_per_micro:
                    skipped.append((q0, q1))
                    continue

                best = None
                best_cost = float("inf")
                p0 = current_pos[q0]
                p1 = current_pos[q1]

                for (a_phys, b_phys, a_grid, b_grid) in pair_phys:
                    a_col = _phys_to_xy_col(a_phys, spacing_xy)
                    b_col = _phys_to_xy_col(b_phys, spacing_xy)
                    if a_col in used_xy_cols or b_col in used_xy_cols:
                        continue
                    if a_col == b_col:
                        continue
                    if a_phys in used_sites or b_phys in used_sites:
                        continue

                    c1 = _euclidean3(p0, a_phys) + _euclidean3(p1, b_phys)
                    c2 = _euclidean3(p0, b_phys) + _euclidean3(p1, a_phys)
                    if c1 < best_cost:
                        best_cost = c1
                        best = (a_phys, b_phys, a_col, b_col, a_grid, b_grid, False)
                    if c2 < best_cost:
                        best_cost = c2
                        best = (b_phys, a_phys, b_col, a_col, b_grid, a_grid, True)

                if best is None:
                    skipped.append((q0, q1))
                    continue

                q0_site, q1_site, q0_col, q1_col, q0_grid, q1_grid, swapped = best
                placements[q0] = q0_site
                placements[q1] = q1_site
                used_xy_cols.add(q0_col)
                used_xy_cols.add(q1_col)
                used_sites.add(q0_site)
                used_sites.add(q1_site)
                stage_travel += _euclidean3(current_pos[q0], q0_site) + _euclidean3(current_pos[q1], q1_site)
                gate_sites.append(
                    {
                        "gate": [q0, q1],
                        "sites_grid": [list(q0_grid), list(q1_grid)],
                        "sites_phys": [list(q0_site), list(q1_site)],
                        "swapped": bool(swapped),
                    }
                )
                assigned.append((q0, q1))

            if not assigned:
                # If we can't place even one gate, it is fundamentally infeasible under current constraints.
                raise RuntimeError(
                    f"[Error] No feasible interaction placement even after micro-stage split. "
                    f"original_stage={orig_stage_idx}, remaining_gate={remaining[0]!r}."
                )

            # Apply placements (qubits used in this micro-stage are now in interaction)
            for q, p in placements.items():
                current_pos[q] = p

            # Remove assigned gates from remaining; keep skipped.
            remaining = [g for g in remaining if g not in assigned]

            # Optional: readout moves only after a qubit finishes ALL 2Q in this original stage AND its global last2q is this stage.
            readout_moves: List[Dict] = []
            if enable_readout_move and last_stage_2q is not None:
                # Build quick lookup: which qubits still have pending 2Q gates in this original stage.
                pending_qubits = set()
                for (rq0, rq1) in remaining:
                    pending_qubits.add(rq0)
                    pending_qubits.add(rq1)
                for q in placements.keys():
                    if q in pending_qubits:
                        continue
                    if 0 <= q < len(last_stage_2q) and int(last_stage_2q[q]) == int(orig_stage_idx):
                        x, y, z = current_pos[q]
                        readout_p = (float(x), float(y), float(readout_plane_z))
                        dz = abs(readout_p[2] - z)
                        stage_travel += dz
                        current_pos[q] = readout_p
                        readout_moves.append({"qubit": int(q), "from": [x, y, z], "to": [readout_p[0], readout_p[1], readout_p[2]]})

            total_travel += stage_travel

            stage_positions.append({int(q): current_pos[q] for q in range(n_qubits)})
            stage_meta.append(
                {
                    "micro_stage": int(micro_stage_idx),
                    "original_stage": int(orig_stage_idx),
                    "micro_in_original": int(micro_in_original),
                    "two_qubit_gates": [list(g) for g in assigned],
                    "gate_sites": gate_sites,
                    "readout_moves": readout_moves,
                    "travel_distance": float(stage_travel),
                    "max_gates_per_micro": int(max_gates_per_micro),
                }
            )
            micro_stage_idx += 1
            micro_in_original += 1

    summary = {
        "total_travel_distance": float(total_travel),
        "n_scheduler_stages": int(len(stages)),
        "n_micro_stages": int(len(stage_positions)),
        "interaction_cols_capacity": int(interaction_cols_capacity),
        "max_gates_per_micro": int(max_gates_per_micro),
    }
    return stage_positions, stage_meta, summary

