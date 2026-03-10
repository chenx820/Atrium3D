import time
from typing import Dict, List, Sequence, Tuple


class Scheduler:
    """ASAP scheduler for a flat gate list g_q."""

    def __init__(self, results_code: Dict, g_q: Sequence[Tuple[int, int]]) -> None:
        self.g_q = g_q
        self.results_code = results_code
        self.results_code["n_stages"] = 0
        self.list_scheduling: List[List[int]] = []


    def get_list_gates(self) -> List[List[int]]:
        list_gate = []
        for stage, gates in enumerate(self.list_scheduling):
            tmp = [self.g_q[i] for i in gates]
            print(f"    Stage {stage}: {tmp}")
            list_gate.append(tmp)
        return list_gate

    def asap(self) -> List[List[int]]:
        """
        Implements As Soon As Possible (ASAP) scheduling for gates.

        Returns:
            List[List[int]]: List of scheduled gate indices per stage.
        """
        print(f"[INFO] Atrium3D: Start ASAP scheduling")
        t0 = time.perf_counter()
        n_qubits = int(self.results_code.get("n_qubits", 0))
        if n_qubits <= 0:
            raise ValueError(
                "[Error] results_code['n_qubits'] must be a positive integer before scheduling."
            )

        list_qubit_stage = [0 for _ in range(n_qubits)]
        for i, gate in enumerate(self.g_q):
            q0, q1 = gate
            if not (0 <= q0 < n_qubits and 0 <= q1 < n_qubits):
                raise ValueError(
                    f"[Error] Gate index out of range: ({q0}, {q1}) with n_qubits={n_qubits}"
                )

            stage0 = list_qubit_stage[q0]
            stage1 = list_qubit_stage[q1]
            stage = max(stage0, stage1)
            if stage >= len(self.list_scheduling):
                self.list_scheduling.append([])
            self.list_scheduling[stage].append(i)

            stage += 1
            list_qubit_stage[q0] = stage
            list_qubit_stage[q1] = stage

        self.results_code["n_stages"] = len(self.list_scheduling)

        print("[INFO] Atrium3D: Scheduling finished")
        elapsed = time.perf_counter() - t0
        self.results_code["compilation_time"]["scheduling"] = float(elapsed)
        self.results_code["compilation_time"]["total"] += float(elapsed)