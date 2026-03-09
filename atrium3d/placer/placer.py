import math
from typing import Dict, List, Optional, Sequence, Tuple

Point3 = Tuple[float, float, float]
Gate = Tuple[int, int]

# physical constant: distance between two qubits for two-qubit gate (um)
GATE_DISTANCE = 2.5
# physical constant: minimum safe distance between two beams/two atoms (um)
MIN_BEAM_DIST = 1.5

class Placer:
    def __init__(self, size: int, layers: int, center_range: range, spacing_xy: float, spacing_z: float):
        self.size = size
        self.layers = layers
        self.center_range = center_range
        self.spacing_xy = spacing_xy
        self.spacing_z = spacing_z


    def _euclidean3(self, a: Point3, b: Point3) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


    def _grid_to_phys(self, site: Tuple[int, int, int]) -> Point3:
        x, y, z = site
        return (float(x * self.spacing_xy), float(y * self.spacing_xy), float(z * self.spacing_z))


    def _interaction_sites_grid(self) -> List[Tuple[int, int, int]]:
        sites = []
        for z in range(self.layers):
            if z == self.layers - 1:
                continue
            for x in self.center_range:
                for y in self.center_range:
                    sites.append((x, y, z))
        return sites


    def place_stages(
        self,
        *,
        initial_mapping: Dict[int, Point3],
        stages: Sequence[Sequence[Gate]],
        last_stage_2q: Optional[Sequence[int]] = None,
        enable_readout_move: bool = True,
    ) -> Tuple[List[Dict[int, Point3]], List[Dict], Dict]:
        """
        Decide the placement of qubits (atoms) for each stage.
        Optimization: introduced Grid-Free Interaction Targeting via AOD Hovering.
        """
        n_qubits = len(initial_mapping)
        current_pos: Dict[int, Point3] = {int(q): (float(p[0]), float(p[1]), float(p[2])) for q, p in initial_mapping.items()}

        # get all SLM anchor sites (Anchor Sites) in the computation region
        interaction_grids = self._interaction_sites_grid()
        anchor_phys_list = [(self._grid_to_phys(g), g) for g in interaction_grids]

        readout_plane_z = float((self.layers - 1) * self.spacing_z)

        stage_positions: List[Dict[int, Point3]] = []
        stage_meta: List[Dict] = []
        total_travel = 0.0

        # after introducing hovering, each gate only occupies 1 anchor point, capacity is significantly increased
        interaction_cols_capacity = len(self.center_range) * len(self.center_range)
        max_gates_per_micro = max(1, interaction_cols_capacity)

        micro_stage_idx = 0

        for orig_stage_idx, stage in enumerate(stages):
            remaining_gates: List[Gate] = [(int(q0), int(q1)) for (q0, q1) in stage if int(q0) != int(q1)]

            def _gate_difficulty(g: Gate) -> float:
                q0, q1 = g
                p0 = current_pos[q0]
                p1 = current_pos[q1]
                cx = float((self.center_range.start + self.center_range.stop - 1) * 0.5 * self.spacing_xy)
                cy = float((self.center_range.start + self.center_range.stop - 1) * 0.5 * self.spacing_xy)
                cz = float((self.layers - 2) * self.spacing_z)
                return self._euclidean3(p0, (cx, cy, cz)) + self._euclidean3(p1, (cx, cy, cz))

            if not remaining_gates:
                stage_positions.append({int(q): current_pos[q] for q in range(n_qubits)})
                stage_meta.append({
                    "micro_stage": int(micro_stage_idx),
                    "original_stage": int(orig_stage_idx),
                    "micro_in_original": 0,
                    "two_qubit_gates": [],
                    "gate_sites": [],
                    "readout_moves": [],
                    "travel_distance": 0.0,
                    "max_gates_per_micro": int(max_gates_per_micro),
                })
                micro_stage_idx += 1
                continue

            micro_in_original = 0

            while remaining_gates:
                remaining_gates.sort(key=_gate_difficulty, reverse=True)
                candidate_order = remaining_gates[:]

                # Find active qubits in this micro-stage
                active_q_in_micro = set()
                for q0, q1 in candidate_order[:max_gates_per_micro]:
                    active_q_in_micro.add(q0)
                    active_q_in_micro.add(q1)

                target_pos = {}
                readout_moves_this_micro = []

                # dynamic column clearance: idle atoms return to Storage or move up to Readout
                for q in range(n_qubits):
                    if q not in active_q_in_micro:
                        is_finished = False
                        if enable_readout_move and last_stage_2q is not None:
                            if last_stage_2q[q] == -1 or orig_stage_idx >= last_stage_2q[q]:
                                is_finished = True

                        base_p = initial_mapping[q]
                        if is_finished:
                            rp = (base_p[0], base_p[1], readout_plane_z)
                            target_pos[q] = rp
                            if current_pos[q] != rp:
                                readout_moves_this_micro.append(
                                    {"qubit": int(q), "from": list(current_pos[q]), "to": list(rp)}
                                )
                        else:
                            target_pos[q] = base_p

                assigned_gates = []
                gate_sites = []
                
                # use continuous physical coordinates to track the occupied beam columns and spatial positions
                used_xy_cols_phys: List[Tuple[float, float]] = []
                used_sites_phys: List[Point3] = []
                
                for p in target_pos.values():
                    used_xy_cols_phys.append((p[0], p[1]))
                    used_sites_phys.append(p)
                
                z_layer_usage = {z: 0 for z in range(self.layers - 1)}

                for (q0, q1) in candidate_order:
                    if len(assigned_gates) >= max_gates_per_micro:
                        continue

                    best = None
                    best_cost = float("inf")
                    p0 = current_pos[q0]
                    p1 = current_pos[q1]

                    for anchor_phys, anchor_grid in anchor_phys_list:
                        # AOD hovering mechanism: control atoms to be held by AOD at a distance from the anchor point GATE_DISTANCE
                        hover_phys = (anchor_phys[0] + GATE_DISTANCE, anchor_phys[1], anchor_phys[2])
                        hover_grid = (anchor_grid[0] + GATE_DISTANCE / self.spacing_xy, anchor_grid[1], anchor_grid[2])
                        
                        col_anchor = (anchor_phys[0], anchor_phys[1])
                        col_hover = (hover_phys[0], hover_phys[1])

                        # continuous space Skewer/physical collision detection
                        conflict = False
                        for used_col in used_xy_cols_phys:
                            if math.hypot(col_anchor[0] - used_col[0], col_anchor[1] - used_col[1]) < MIN_BEAM_DIST:
                                conflict = True; break
                            if math.hypot(col_hover[0] - used_col[0], col_hover[1] - used_col[1]) < MIN_BEAM_DIST:
                                conflict = True; break
                        if conflict: continue

                        for used_p in used_sites_phys:
                            if self._euclidean3(anchor_phys, used_p) < MIN_BEAM_DIST:
                                conflict = True; break
                            if self._euclidean3(hover_phys, used_p) < MIN_BEAM_DIST:
                                conflict = True; break
                        if conflict: continue

                        # calculate the cost of transportation
                        c1 = self._euclidean3(p0, anchor_phys) + self._euclidean3(p1, hover_phys)
                        c2 = self._euclidean3(p0, hover_phys) + self._euclidean3(p1, anchor_phys)

                        z_idx = anchor_grid[2]
                        z_penalty = z_layer_usage[z_idx] * 50.0  # Z-axis congestion decoupling penalty

                        if c1 + z_penalty < best_cost:
                            best_cost = c1 + z_penalty
                            best = (anchor_phys, hover_phys, col_anchor, col_hover, anchor_grid, hover_grid, False, z_idx)
                        if c2 + z_penalty < best_cost:
                            best_cost = c2 + z_penalty
                            best = (hover_phys, anchor_phys, col_hover, col_anchor, hover_grid, anchor_grid, True, z_idx)

                    if best is not None:
                        a_phys, b_phys, col_a, col_b, grid_a, grid_b, swapped, z_idx = best
                        target_pos[q0] = a_phys
                        target_pos[q1] = b_phys
                        
                        used_xy_cols_phys.append(col_a)
                        used_xy_cols_phys.append(col_b)
                        used_sites_phys.append(a_phys)
                        used_sites_phys.append(b_phys)
                        z_layer_usage[z_idx] += 1
                        
                        assigned_gates.append((q0, q1))
                        gate_sites.append({
                            "gate": [q0, q1],
                            # hover_grid has floating-point numbers, but can be converted to a list for JSON serialization
                            "sites_grid": [list(grid_a), list(grid_b)],
                            "sites_phys": [list(a_phys), list(b_phys)],
                            "swapped": bool(swapped),
                        })

                if not assigned_gates:
                    raise RuntimeError(
                        f"[Error] Deadlock in micro-stage placement! "
                        f"original_stage={orig_stage_idx}, remaining_gates={remaining_gates}"
                    )

                # update physical positions and accumulate the real transportation distance
                stage_travel = 0.0
                for q in range(n_qubits):
                    if q in target_pos:
                        dist = self._euclidean3(current_pos[q], target_pos[q])
                        stage_travel += dist
                        current_pos[q] = target_pos[q]

                total_travel += stage_travel

                stage_positions.append({int(q): current_pos[q] for q in range(n_qubits)})
                stage_meta.append({
                    "micro_stage": int(micro_stage_idx),
                    "original_stage": int(orig_stage_idx),
                    "micro_in_original": int(micro_in_original),
                    "two_qubit_gates": [list(g) for g in assigned_gates],
                    "gate_sites": gate_sites,
                    "readout_moves": readout_moves_this_micro,
                    "travel_distance": float(stage_travel),
                    "max_gates_per_micro": int(max_gates_per_micro),
                })

                micro_stage_idx += 1
                micro_in_original += 1
                remaining_gates = [g for g in remaining_gates if g not in assigned_gates]

        # final positioning
        final_target_pos = {}
        final_readout_moves = []
        final_travel = 0.0
        for q in range(n_qubits):
            base_p = initial_mapping[q]
            if enable_readout_move:
                rp = (base_p[0], base_p[1], readout_plane_z)
                final_target_pos[q] = rp
                if current_pos[q] != rp:
                    final_readout_moves.append({"qubit": int(q), "from": list(current_pos[q]), "to": list(rp)})
            else:
                final_target_pos[q] = base_p

        if final_readout_moves or any(current_pos[q] != final_target_pos[q] for q in range(n_qubits)):
            for q in range(n_qubits):
                final_travel += self._euclidean3(current_pos[q], final_target_pos[q])
                current_pos[q] = final_target_pos[q]
            total_travel += final_travel

            stage_positions.append({int(q): current_pos[q] for q in range(n_qubits)})
            stage_meta.append({
                "micro_stage": int(micro_stage_idx),
                "original_stage": int(len(stages)),
                "micro_in_original": 0,
                "two_qubit_gates": [],
                "gate_sites": [],
                "readout_moves": final_readout_moves,
                "travel_distance": float(final_travel),
                "max_gates_per_micro": int(max_gates_per_micro),
            })

        summary = {
            "total_travel_distance": float(total_travel),
            "n_scheduler_stages": int(len(stages)),
            "n_micro_stages": int(len(stage_positions)),
            "interaction_cols_capacity": int(interaction_cols_capacity),
            "max_gates_per_micro": int(max_gates_per_micro),
        }
        return stage_positions, stage_meta, summary