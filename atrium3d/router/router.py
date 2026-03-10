from typing import Dict, List, Tuple, Sequence
from atrium3d.placer.placer import InitialPlacer
from atrium3d.placer.placer import StagePlacer

Point3 = Tuple[float, float, float]

class Router:
    def __init__(
        self,
        results_code: Dict,
        sites: Sequence[Sequence[Point3]],
        list_full_gates: Sequence[Sequence[Tuple[int, int]]],
        initial_mapping: Dict[int, Point3]=None,
    ):
        self.results_code = results_code
        self.storage_zone, self.interaction_zone, self.readout_zone = sites
        self.list_full_gates = list_full_gates
        self.n_q = int(self.results_code.get("n_qubits", 0))
        if initial_mapping is not None:
            self.initial_mapping = dict(initial_mapping)
        else:
            self.initial_mapping = None

        # Simple timing placeholders (not modeled in this repo yet).
        self.time_1q_gate = 0
        self.time_2q_gate = 0
        self.stage_index = 0


    def write_init_instruction(self):
        """
        Writes initialization instructions for qubit mapping.
        """
        self.results_code['instructions'].clear()
        self.results_code['instructions'].append(
            {
                'type': "Init",
                'duration': 0,
                'locs': [{
                    'id': q,
                    'x': self.current_mapping[q][0],
                    'y': self.current_mapping[q][1],
                    'z': self.current_mapping[q][2]
                } for q in range(self.n_q)]
            }
        )


    
    def write_1q_gate_instruction(self, gate_1q: list):
        """
        Writes instructions for single-qubit gates.

        Args:
            gate_1q (list): List of qubits for single-qubit gates.
        """
        locs = [
            {
                'id': q,
                'x': self.current_mapping[q][0],
                'y': self.current_mapping[q][1]
            } for q in gate_1q
        ]
        self.results_code['instructions'].append({
            'type': "1qGate",
            'stage': self.stage_index,
            'duration': self.time_1q_gate,
            'qs': gate_1q,
            'gates': gate_1q,
            'locs': locs
        })

    def write_2q_gate_instruction(self, gate_2q: list):
        """
        Writes instructions for two-qubit gates.

        Args:
            gate_2q (tuple): Tuple of qubit pairs for two-qubit gates.
        """
        locs, qs = [], []
        for q0, q1 in gate_2q:
            qs += [q0, q1]
            locs.extend([{
                'id': q,
                'x': self.current_mapping[q][0],
                'y': self.current_mapping[q][1],
                'z': self.current_mapping[q][2]
            } for q in [q0, q1]])

        self.results_code['instructions'].append({
            'type': "2qGate",
            'stage': self.stage_index,
            'duration': self.time_2q_gate,
            'qs': qs,
            'gates': gate_2q,
            'locs': locs
        })

    def route_qubits(self):
        placer = InitialPlacer(
            results_code=self.results_code,
            sites=[self.storage_zone, self.interaction_zone, self.readout_zone],
            list_full_gates=self.list_full_gates,
            )
        self.current_mapping = placer.solve()
        self.write_init_instruction()

        for self.stage_idx, gates in enumerate(self.list_full_gates):
            placer = StagePlacer(
                results_code=self.results_code,
                sites=[self.storage_zone, self.interaction_zone, self.readout_zone],
                list_full_gates=self.list_full_gates[self.stage_idx:],
                initial_mapping=self.current_mapping,
            )
            self.current_mapping = placer.solve()
            
            placer = StagePlacer(
                results_code=self.results_code,
                sites=[self.storage_zone, self.interaction_zone, self.readout_zone],
                list_full_gates=self.list_full_gates[self.stage_idx:],
                initial_mapping=self.current_mapping,
            )
            self.current_mapping = placer.solve()
            
            gates_1q = []
            gates_2q = []
            for q0, q1 in gates:
                if q0 == q1:
                    # 处理单比特门
                    gates_1q.append(q0)
                else:
                    gates_2q.append((q0, q1))
                    # 处理双比特门
                
            if gates_1q:
                self.write_1q_gate_instruction(gates_1q)
            if gates_2q:
                self.write_2q_gate_instruction(gates_2q)

            self.initial_mapping = self.current_mapping