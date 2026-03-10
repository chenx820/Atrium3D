import math
import random
import time
import sys
from typing import Dict, List, Optional, Sequence, Tuple

Point3 = Tuple[float, float, float]
Gate = Tuple[int, int]

# physical constant: distance between two qubits for two-qubit gate (um)
GATE_DISTANCE = 2.5
# physical constant: minimum safe distance between two beams/two atoms (um)
MIN_BEAM_DIST = 1.5

# SA penalty weight for AOD segment collisions (initial placement)
VIOLATION_WEIGHT = 10000.0


def _ensure_compilation_time(results_code: Dict) -> Dict:
    ct = results_code.setdefault("compilation_time", {})
    ct.setdefault("total", 0.0)
    ct.setdefault("initial_mapping", 0.0)
    return results_code


def _euclidean3(a: Point3, b: Point3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def segment_distance_3d(p1: Point3, p2: Point3, p3: Point3, p4: Point3) -> float:
    """Shortest distance between two 3D segments (p1->p2 and p3->p4)."""
    u = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    v = (p4[0] - p3[0], p4[1] - p3[1], p4[2] - p3[2])
    w = (p1[0] - p3[0], p1[1] - p3[1], p1[2] - p3[2])

    a = sum(x * y for x, y in zip(u, u))
    b = sum(x * y for x, y in zip(u, v))
    c = sum(x * y for x, y in zip(v, v))
    d = sum(x * y for x, y in zip(u, w))
    e = sum(x * y for x, y in zip(v, w))

    D = a * c - b * b
    sN = D
    sD = D
    tN = D
    tD = D

    if D < 1e-8:
        sN, sD = 0.0, 1.0
        tN, tD = e, c
    else:
        sN = b * e - c * d
        tN = a * e - b * d
        if sN < 0.0:
            sN, tN, tD = 0.0, e, c
        elif sN > sD:
            sN, tN, tD = sD, e + b, c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN, sD = -d, a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN, sD = (-d + b), a

    sc = 0.0 if abs(sN) < 1e-8 else sN / sD
    tc = 0.0 if abs(tN) < 1e-8 else tN / tD

    dP = (
        w[0] + sc * u[0] - tc * v[0],
        w[1] + sc * u[1] - tc * v[1],
        w[2] + sc * u[2] - tc * v[2],
    )
    return math.sqrt(sum(x * x for x in dP))


class Initial3DPlacer:
    """Simulated Annealing (SA) based 3D initial placer for Atrium."""

    def __init__(
        self,
        sites: Sequence[Point3],
        n_qubits: int,
        list_full_gates: Sequence[Sequence[Gate]],
        readout_plane_z: Optional[float] = None,
        readout_urgency: Optional[Sequence[float]] = None,
        readout_weight: float = 0.0,
        seed_val: int = 0,
    ):
        random.seed(seed_val)
        self.sites = sites
        self.n_qubits = n_qubits
        self.list_full_gates = list_full_gates
        self.n_sites = len(sites)

        # partitioning architecture params
        self.readout_plane_z = readout_plane_z
        self.readout_urgency = readout_urgency
        self.readout_weight = readout_weight

        # precompute gates per qubit
        self.qubit_to_gates = [[] for _ in range(self.n_qubits)]
        for stage_idx, stage in enumerate(self.list_full_gates):
            for gate_idx, (q0, q1) in enumerate(stage):
                if q0 != q1:
                    self.qubit_to_gates[q0].append((stage_idx, gate_idx, q1))
                    self.qubit_to_gates[q1].append((stage_idx, gate_idx, q0))

        # SA params
        self.sa_t = 10000.0
        self.sa_t_frozen = 0.001
        self.sa_l = 200
        self.sa_n = 0
        self.sa_iter_limit = 5000

        self.current_mapping = [-1] * self.n_qubits  # qubit -> site_idx
        self.site_to_qubit = [-1] * self.n_sites  # site_idx -> qubit
        self.best_mapping: List[int] = []
        self.current_cost = sys.maxsize
        self.best_cost = sys.maxsize

    def init_solution(self) -> None:
        available_sites = list(range(self.n_sites))
        random.shuffle(available_sites)
        for q in range(self.n_qubits):
            s = available_sites[q]
            self.current_mapping[q] = s
            self.site_to_qubit[s] = q
        self.current_cost = self.get_total_cost()
        self.best_cost = self.current_cost
        self.best_mapping = self.current_mapping[:]

    def run(self) -> Dict[int, Point3]:
        self.init_solution()
        print("[INFO] Atrium3D-3D: Start SA-based 3D initial placement")
        while self.sa_t > self.sa_t_frozen and self.sa_n < self.sa_iter_limit:
            self.sa_n += 1
            accept_count = 0
            for _ in range(self.sa_l):
                q1 = random.randrange(self.n_qubits)
                s1 = self.current_mapping[q1]
                s2 = random.randrange(self.n_sites)
                if s1 == s2:
                    continue
                q2 = self.site_to_qubit[s2]
                delta = self.calculate_delta(q1, s1, q2, s2)
                if delta <= 0 or random.random() <= math.exp(-delta / self.sa_t):
                    # accept
                    self.current_mapping[q1] = s2
                    self.site_to_qubit[s2] = q1
                    self.site_to_qubit[s1] = q2
                    if q2 != -1:
                        self.current_mapping[q2] = s1
                    self.current_cost += delta
                    if self.current_cost < self.best_cost:
                        self.best_cost = self.current_cost
                        self.best_mapping = self.current_mapping[:]
                    accept_count += 1

            # cooling
            if accept_count == 0:
                self.sa_t *= 0.8
            else:
                self.sa_t *= 0.95

        return {q: self.sites[self.best_mapping[q]] for q in range(self.n_qubits)}

    def eval_qubit_cost(self, q: int, site_idx: int) -> float:
        if q == -1:
            return 0.0
        cost = 0.0
        p1 = self.sites[site_idx]
        # interaction distance cost
        for (stage_idx, _gate_idx, peer_q) in self.qubit_to_gates[q]:
            peer_site_idx = self.current_mapping[peer_q]
            if peer_site_idx != -1:
                p2 = self.sites[peer_site_idx]
                weight = max(1 - 0.05 * stage_idx, 0.1)
                cost += weight * _euclidean3(p1, p2)
        # readout pull
        if self.readout_weight > 0 and self.readout_plane_z is not None and self.readout_urgency is not None:
            dz = max(0.0, float(self.readout_plane_z) - float(p1[2]))
            cost += float(self.readout_weight) * float(self.readout_urgency[q]) * dz
        return cost

    def compatible_2D(self, a, b) -> bool:
        """Check if two projected segments are compatible (no crossing) in 2D."""
        # column compatibility
        if a[0] == b[0] and a[1] != b[1]:
            return False
        if a[1] == b[1] and a[0] != b[0]:
            return False
        if a[0] < b[0] <= a[1]:
            return False
        if a[0] > b[0] >= a[1]:
            return False
        # row compatibility
        if a[2] == b[2] and a[3] != b[3]:
            return False
        if a[3] == b[3] and a[2] != b[2]:
            return False
        if a[2] < b[2] <= a[3]:
            return False
        if a[2] > b[2] >= a[3]:
            return False
        return True

    def count_aod_violations(self, mapping_dict: Sequence[int]) -> int:
        """Count global AOD segment collisions under current mapping."""
        violations = 0
        for stage in self.list_full_gates:
            gates_in_stage = [g for g in stage if g[0] != g[1]]
            for i in range(len(gates_in_stage)):
                for j in range(i + 1, len(gates_in_stage)):
                    q1, q2 = gates_in_stage[i]
                    q3, q4 = gates_in_stage[j]
                    p1 = self.sites[mapping_dict[q1]]
                    p2 = self.sites[mapping_dict[q2]]
                    p3 = self.sites[mapping_dict[q3]]
                    p4 = self.sites[mapping_dict[q4]]
                    # x‑AOD view (y,z)
                    vec1 = [p1[1], p1[2], p2[1], p2[2]]
                    vec2 = [p3[1], p3[2], p4[1], p4[2]]
                    # y‑AOD view (x,z)
                    vec3 = [p1[0], p1[2], p3[0], p3[2]]
                    vec4 = [p2[0], p2[2], p4[0], p4[2]]
                    if not self.compatible_2D(vec1, vec2) or not self.compatible_2D(vec3, vec4):
                        violations += 1
        return violations

    def get_total_cost(self) -> float:
        cost = sum(self.eval_qubit_cost(q, self.current_mapping[q]) for q in range(self.n_qubits)) / 2.0
        # readout pull cost (not halved)
        readout_c = 0.0
        if self.readout_weight > 0 and self.readout_plane_z is not None and self.readout_urgency is not None:
            for q in range(self.n_qubits):
                p = self.sites[self.current_mapping[q]]
                dz = max(0.0, float(self.readout_plane_z) - float(p[2]))
                readout_c += float(self.readout_weight) * float(self.readout_urgency[q]) * dz
        cost += readout_c
        cost += self.count_aod_violations(self.current_mapping) * VIOLATION_WEIGHT
        return cost

    def calculate_delta(self, q1: int, s1: int, q2: int, s2: int) -> float:
        """Cost delta for swapping two sites."""
        old_cost = self.eval_qubit_cost(q1, s1) + self.eval_qubit_cost(q2, s2)
        if q2 != -1 and q1 != -1:
            for (stage_idx, _gate_idx, peer_q) in self.qubit_to_gates[q1]:
                if peer_q == q2:
                    weight = max(1 - 0.05 * stage_idx, 0.1)
                    old_cost -= weight * _euclidean3(self.sites[s1], self.sites[s2]) * 2

        old_violations = self.count_aod_violations(self.current_mapping)

        self.current_mapping[q1] = s2
        if q2 != -1:
            self.current_mapping[q2] = s1

        new_cost = self.eval_qubit_cost(q1, s2) + self.eval_qubit_cost(q2, s1)
        if q2 != -1 and q1 != -1:
            for (stage_idx, _gate_idx, peer_q) in self.qubit_to_gates[q1]:
                if peer_q == q2:
                    weight = max(1 - 0.05 * stage_idx, 0.1)
                    new_cost -= weight * _euclidean3(self.sites[s2], self.sites[s1]) * 2

        new_violations = self.count_aod_violations(self.current_mapping)

        # restore mapping
        self.current_mapping[q1] = s1
        if q2 != -1:
            self.current_mapping[q2] = s2

        delta = (new_cost - old_cost) + (new_violations - old_violations) * VIOLATION_WEIGHT
        return delta


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
    """Top-level initial 3D placement function (SA-based)."""
    _ensure_compilation_time(results_code)
    t0 = time.perf_counter()

    n_qubits = int(results_code.get("n_qubits", 0))
    if n_qubits <= 0:
        raise ValueError("[Error] results_code['n_qubits'] must be set before placing_3d().")

    sites = list(available_sites)
    if len(sites) < n_qubits:
        raise ValueError(f"[Error] #qubits ({n_qubits}) > #available_sites ({len(sites)}).")

    # If the user provided a fixed mapping, use it directly
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
                    idx = int(qubits_mapping[q])
                    used.add(idx)
                    mapping_dict[q] = sites[idx]

        results_code["initial_mapping"] = mapping_dict
        elapsed = time.perf_counter() - t0
        results_code["compilation_time"]["initial_mapping"] = float(elapsed)
        results_code["compilation_time"]["total"] += float(elapsed)
        results_code["initial_mapping_space"] = "3d"
        return results_code, mapping_dict

    placer = Initial3DPlacer(
        sites=sites,
        n_qubits=n_qubits,
        list_full_gates=list_full_gates,
        readout_plane_z=readout_plane_z,
        readout_urgency=readout_urgency,
        readout_weight=readout_weight,
        seed_val=0,
    )

    best_mapping = placer.run()

    results_code["initial_mapping"] = best_mapping
    results_code["initial_mapping_space"] = "3d"

    elapsed = time.perf_counter() - t0
    results_code["compilation_time"]["initial_mapping"] = float(elapsed)
    results_code["compilation_time"]["total"] += float(elapsed)

    return results_code, best_mapping


class StagePlacer:
    """Per-stage / per-micro-stage placer on the 3D Atrium architecture.

    Given an initial 3D mapping and a scheduled list of stages, this class decides
    where each qubit should be for every micro‑stage, including interaction/hover
    positions and readout moves.
    """

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