from __future__ import annotations

import time
from typing import Dict, Iterable, List, Sequence, Tuple

class Scheduler:
    def __init__(self,
                 g_q: Sequence[Tuple[int, int]],
                 results_code: Dict,
                 ):
        """
        Initializes the Scheduler class.

        Args:
            g_q (list): List of gates.
            results_code (dict): Results dictionary.
        """
        self.g_q = g_q
        self.results_code = results_code
        self.results_code['n_stages'] = 0
        self.list_scheduling = []

    def asap(self):
        """
        Implements As Soon As Possible (ASAP) scheduling for gates. 

        Returns:
            list: List of scheduled gates at each stage.
        """

        n_qubits = int(self.results_code.get('n_qubits', 0))
        if n_qubits <= 0:
            raise ValueError("[Error] results_code['n_qubits'] must be a positive integer before scheduling.")

        list_qubit_stage = [0 for _ in range(n_qubits)]
        for i, gate in enumerate(self.g_q):
            q0, q1 = gate
            if not (0 <= q0 < n_qubits and 0 <= q1 < n_qubits):
                raise ValueError(f"[Error] Gate index out of range: ({q0}, {q1}) with n_qubits={n_qubits}")

            stage0 = list_qubit_stage[q0]
            stage1 = list_qubit_stage[q1]
            stage = max(stage0, stage1)
            if stage >= len(self.list_scheduling):
                self.list_scheduling.append([])
            self.list_scheduling[stage].append(i)

            stage += 1
            list_qubit_stage[q0] = stage
            list_qubit_stage[q1] = stage

        self.results_code['n_stages'] = len(self.list_scheduling)

        return self.list_scheduling


def _ensure_compilation_time(results_code: Dict) -> Dict:
    ct = results_code.setdefault('compilation_time', {})
    ct.setdefault('total', 0.0)
    ct.setdefault('scheduling', 0.0)
    return results_code


def scheduling(
    g_q: Sequence[Tuple[int, int]],
    results_code: Dict,
    scheduling_strategy: str = "asap",
) -> Tuple[Dict, List[List[int]]]:
    """
    Scheduling the gates.
    """
    _ensure_compilation_time(results_code)
    t0 = time.perf_counter()
    scheduler = Scheduler(g_q=g_q, results_code=results_code)
    strategy = (scheduling_strategy or "asap").lower()
    if strategy == "asap":
        list_scheduling = scheduler.asap()
    else:
        raise ValueError(f"[Error] Unsupported scheduling strategy: {scheduling_strategy!r}")

    elapsed = time.perf_counter() - t0
    results_code['compilation_time']['scheduling'] = float(elapsed)
    results_code['compilation_time']['total'] = float(results_code['compilation_time'].get('total', 0.0) + elapsed)

    return results_code, list_scheduling