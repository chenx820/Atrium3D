import numpy as np
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

# SA penalty weight for AOD violations
VIOLATION_WEIGHT = 10000.0


def _euclidean3(a: Point3, b: Point3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _compatible_2D(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
        # check for column compatibility
        if a[0] == b[0] and a[1] != b[1]:
            return False
        if a[1] == b[1] and a[0] != b[0]:
            return False
        if a[0] < b[0] and a[1] >= b[1]:
            return False
        if a[0] > b[0] and a[1] <= b[1]:
            return False

        # Check for row compatibility
        if a[2] == b[2] and a[3] != b[3]:
            return False
        if a[3] == b[3] and a[2] != b[2]:
            return False
        if a[2] < b[2] and a[3] >= b[3]:
            return False
        if a[2] > b[2] and a[3] <= b[3]:
            return False

        return True


class InitialPlacer:
    """Simulated Annealing (SA) based 3D initial placer for Atrium3D."""

    def __init__(
        self,
        results_code: Dict,
        sites: Sequence[Sequence[Point3]],
        list_full_gates: Sequence[Sequence[Gate]],
        ):
        self.results_code = results_code
        if len(sites) != 3:
            raise ValueError(
                "[Error] InitialPlacer expects sites=[storage_zone, interaction_zone, readout_zone]."
            )
        storage_zone, interaction_zone, readout_zone = sites
        self.storage_zone = list(storage_zone)
        self.interaction_zone = list(interaction_zone)
        self.readout_zone = list(readout_zone)
        # Internal SA uses indices; external API returns coordinates.
        self.sites: List[Point3] = self.storage_zone + self.interaction_zone + self.readout_zone
        self.list_full_gates = list_full_gates

        
        self.n_qubits = int(self.results_code.get("n_qubits", 0))
        self.n_sites = len(self.sites)

        # precompute gates per qubit
        self.qubit_to_gates = [[] for _ in range(self.n_qubits)]
        for stage_idx, stage in enumerate(self.list_full_gates):
            for gate_idx, (q0, q1) in enumerate(stage):
                if q0 != q1:
                    self.qubit_to_gates[q0].append((stage_idx, gate_idx, q1))
                    self.qubit_to_gates[q1].append((stage_idx, gate_idx, q0))

        self.current_mapping: List[int] = [-1] * self.n_qubits
        self.site_to_qubit = [-1] * self.n_sites  # site_idx -> qubit
        self.best_mapping: List[int] = [-1] * self.n_qubits

        self._init_sa_params()


    def _init_sa_params(self) -> None:
        # SA params
        self.sa_t = 10000.0
        self.sa_t_frozen = 0.001
        self.sa_l = 200
        self.sa_n = 0
        self.sa_iter_limit = 5000


        self.current_cost = sys.maxsize
        self.best_cost = sys.maxsize


    def _init_solution(self) -> None:
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
        self._init_solution()
        print("[INFO] Atrium3D: Start SA-based 3D initial placement")
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
        if q == -1 or site_idx == -1:
            return 0.0
        cost = 0.0
        p1 = self.sites[site_idx]
        # interaction distance cost
        for (stage_idx, _gate_idx, peer_q) in self.qubit_to_gates[q]:
            peer_site_idx = self.current_mapping[peer_q]
            if peer_site_idx != -1:
                p2 = self.sites[peer_site_idx]
                weight = np.exp(- stage_idx)
                cost += weight * _euclidean3(p1, p2)
        return cost


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

                    if not _compatible_2D(vec1, vec2) or not _compatible_2D(vec3, vec4):
                        violations += 1
        return violations

    def get_total_cost(self) -> float:
        cost = sum(self.eval_qubit_cost(q, self.current_mapping[q]) for q in range(self.n_qubits)) / 2.0
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


    def solve(self) -> Dict[int, Point3]:
        """Run SA and return a qubit->(x,y,z) mapping."""
        t0 = time.perf_counter()

        n_qubits = int(self.results_code.get("n_qubits", 0))
        if n_qubits <= 0:
            raise ValueError(
                "[Error] self.results_code['n_qubits'] must be set before InitialPlacer.solve()."
            )

        if len(self.sites) < n_qubits:
            raise ValueError(
                f"[Error] #qubits ({n_qubits}) > #available_sites ({len(self.sites)})."
            )

        # Run simulated annealing to obtain the best mapping.
        mapping = self.run()

        # Record compilation time.
        elapsed = time.perf_counter() - t0
        self.results_code["compilation_time"]["initial_mapping"] = float(elapsed)
        self.results_code["compilation_time"]["total"] += float(elapsed)

        self.results_code["initial_mapping"] = mapping
        return mapping



class StagePlacer:

    def __init__(
        self,
        results_code: Dict,
        sites: Sequence[Sequence[Point3]],
        list_full_gates: Sequence[Sequence[Gate]],
        initial_mapping: List[int],
        ):
        self.results_code = results_code
        if len(sites) != 3:
            raise ValueError(
                "[Error] InitialPlacer expects sites=[storage_zone, interaction_zone, readout_zone]."
            )
        storage_zone, interaction_zone, readout_zone = sites
        self.storage_zone = list(storage_zone)
        self.interaction_zone = list(interaction_zone)
        self.readout_zone = list(readout_zone)
        # Internal SA uses indices; external API returns coordinates.
        self.sites: List[Point3] = self.storage_zone + self.interaction_zone + self.readout_zone
        self.list_full_gates = list_full_gates

        self.current_mapping = initial_mapping
        self.n_qubits = int(self.results_code.get("n_qubits", 0))
        self.n_sites = len(self.sites)

        # precompute gates per qubit
        self.qubit_to_gates = [[] for _ in range(self.n_qubits)]
        for stage_idx, stage in enumerate(self.list_full_gates):
            for gate_idx, (q0, q1) in enumerate(stage):
                if q0 != q1:
                    self.qubit_to_gates[q0].append((stage_idx, gate_idx, q1))
                    self.qubit_to_gates[q1].append((stage_idx, gate_idx, q0))

        

        self._init_sa_params()


    def _init_sa_params(self) -> None:
        # SA params
        self.sa_t = 10000.0
        self.sa_t_frozen = 0.001
        self.sa_l = 200
        self.sa_n = 0
        self.sa_iter_limit = 5000


        self.current_cost = sys.maxsize
        self.best_cost = sys.maxsize

    def _init_solution(self) -> None:
        self.normal_vectors_x = []
        self.normal_vectors_y = []
        self.reverse_vectors_x = []
        self.reverse_vectors_y = []

        for q0, q1 in self.gates_2q:
            site0 = self.current_mapping[q0]
            site1 = self.current_mapping[q1]
            vec_x = [site0[1], site0[2], site1[1], site1[2]]
            vec_y = [site0[0], site0[2], site1[0], site1[2]]
            reverse_vec_x = [site1[1], site1[2], site0[1], site0[2]]
            reverse_vec_y = [site1[0], site1[2], site0[0], site0[2]]

            self.normal_vectors_x.append(vec_x)
            self.normal_vectors_y.append(vec_y)
            self.reverse_vectors_x.append(reverse_vec_x)
            self.reverse_vectors_y.append(reverse_vec_y)

    def count_aod_violations(
        self,
        vectors_x: Sequence[List[float]],
        vectors_y: Sequence[List[float]],
    ) -> int:
        """Count AOD segment collisions for the current stage.

        `vectors_x` / `vectors_y` are parallel lists, each element being a
        4-tuple-like `[x0, z0, x1, z1]` or `[y0, z0, y1, z1]` describing one
        gate's projected segment. We only need to check pairwise compatibility
        between these segments, independent of `list_full_gates` length.
        """

        violations = 0
        n = min(len(vectors_x), len(vectors_y))
        for i in range(n):
            for j in range(i + 1, n):
                vec1 = vectors_x[i]
                vec2 = vectors_x[j]
                vec3 = vectors_y[i]
                vec4 = vectors_y[j]
                if not _compatible_2D(vec1, vec2) or not _compatible_2D(vec3, vec4):
                    violations += 1
        return violations

    def calculate_delta(self, gate_idx: int) -> float:
        """Cost delta for reversing a gate."""
        old_violations = self.count_aod_violations(self.normal_vectors_x, self.normal_vectors_y)

        self.normal_vectors_x[gate_idx] = self.reverse_vectors_x[gate_idx]
        self.normal_vectors_y[gate_idx] = self.reverse_vectors_y[gate_idx]

        new_violations = self.count_aod_violations(self.normal_vectors_x, self.normal_vectors_y)

        delta = new_violations - old_violations
        return delta

    
    def parking(self, site0: Point3, site1: Point3) -> None:
        """
        Move along the vector between two qubits from site0 to site1 until the distance to site1 is 2,
        and set the site to the stopped coordinate.
        """
        vec = [site1[i] - site0[i] for i in range(3)]
        dist = math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
        if dist == 0:
            # Can't move if initial and target are in same position
            return

        stop_dist = 2.0
        alpha = max((dist - stop_dist) / dist, 0)
        new_site = [site0[i] + vec[i] * alpha for i in range(3)]
        return new_site


    def run(self) -> None:
        self.gates_1q = []
        self.gates_2q = []
        for q0, q1 in self.list_full_gates[0]:
            if q0 == q1:
                self.gates_1q.append(q0)
            else:
                self.gates_2q.append((q0, q1))

        if self.gates_2q:

            self._init_solution()

            vectors_x = self.normal_vectors_x
            vectors_y = self.normal_vectors_y
            while self.sa_t > self.sa_t_frozen and self.sa_n < self.sa_iter_limit:
                self.sa_n += 1
                
                accept_count = 0
                for _ in range(self.sa_l):
                    gate_idx = random.randrange(len(self.gates_2q))
                    vectors_x[gate_idx] = self.reverse_vectors_x[gate_idx]
                    vectors_y[gate_idx] = self.reverse_vectors_y[gate_idx]

                    delta = self.calculate_delta(gate_idx)
                    if delta <= 0 or random.random() <= math.exp(-delta / self.sa_t):
                        # accept
                        accept_count += 1

                # cooling
                if accept_count == 0:
                    self.sa_t *= 0.8
                else:
                    self.sa_t *= 0.95
            for idx, (q0, q1) in enumerate(self.gates_2q):
                site0 = self.current_mapping[q0]
                site1 = self.current_mapping[q1]
                if vectors_x[idx] != self.normal_vectors_x[idx] or vectors_y[idx] != self.normal_vectors_y[idx]:
                    self.current_mapping[q0] = self.parking(site0, site1)
                else:
                    self.current_mapping[q1] = self.parking(site1, site0)


    def solve(self) -> List[int]:
        """Run SA and return a qubit->(x,y,z) mapping."""
        t0 = time.perf_counter()

        # Run simulated annealing to obtain the best mapping.
        self.run()

        # Record compilation time.
        elapsed = time.perf_counter() - t0
        self.results_code["compilation_time"]["stage_mapping"] = float(elapsed)
        self.results_code["compilation_time"]["total"] += float(elapsed)

        return self.current_mapping

