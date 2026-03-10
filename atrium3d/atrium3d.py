import json
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from atrium3d.scheduler.scheduler import Scheduler
from atrium3d.placer.placer import InitialPlacer
from atrium3d.router.router import Router
from atrium3d.animator.animator import Animator

Point3 = Tuple[float, float, float]

class Atrium3D:
    def __init__(
        self,
        benchmark: str,
        dir: str = "default",
        type: str = "qasm",
        size: int = 10,
        layers: int = 4,
        spacing_xy: float = 5.0,
        spacing_z: float = 25.0,
        routing_steps_per_move: int = 15,
        routing_pause_frames: int = 5,
        architecture: Optional[Dict] = None,
        scheduling_strategy: str = "asap",
        given_initial_mapping=None,
    ):
        self.benchmark = benchmark
        self.dir = dir
        self.type = type
        self.size = size
        self.layers = layers
        self.spacing_xy = float(spacing_xy)
        self.spacing_z = float(spacing_z)
        self.routing_steps_per_move = int(routing_steps_per_move)
        self.routing_pause_frames = int(routing_pause_frames)
        self.scheduling_strategy = scheduling_strategy
        self.given_initial_mapping = given_initial_mapping
        self.center_range = range(2, self.size - 2) # center area is the interaction zone

        self.storage_zone = set()
        self.interaction_zone = set()
        self.readout_zone = set()

        # SLM trap sites for placing/routing
        self.qubit_slm_sites: List[Point3] = []
        self.buffer_slm_sites: List[Point3] = []

        self.architecture = architecture or self._default_architecture()

        # zones: Storage (L1-L{layers-1}), Interaction (center), Readout (top plane)
        for x in range(self.size):
            for y in range(self.size):
                for z in range(self.layers):
                    if z == self.layers - 1:
                        zone = "Readout"
                    elif x in self.center_range and y in self.center_range:
                        zone = "Interaction"
                    else:
                        zone = "Storage"
                    self.get_zone(zone, x, y, z)

        self.results_code = {
            'benchmark': benchmark,
            'dir': self.dir,
            'compilation_time': {
                'total': 0.0,
                'scheduling': 0.0,
                'initial_mapping': 0.0,
                'routing': 0.0
            },
            'n_qubits': 0,
            'n_stages': 0,
            'instructions': []
        }

        # Program / gate list
        self.g_q: List[Tuple[int, int]] = []

    def get_phys_pos(self, grid_pos):
        # Store as a hashable 3-tuple so it can live in a set.
        return (
            float(grid_pos[0] * self.spacing_xy),
            float(grid_pos[1] * self.spacing_xy),
            float(grid_pos[2] * self.spacing_z),
        )

    def get_zone(self, zone, x, y, z):
        if zone == "Storage":
            self.storage_zone.add(self.get_phys_pos((x, y, z)))
        elif zone == "Interaction":
            self.interaction_zone.add(self.get_phys_pos((x, y, z)))
        elif zone == "Readout":
            self.readout_zone.add(self.get_phys_pos((x, y, z)))

    def visualize(self, save_path=None):
        fig = plt.figure(figsize=(self.size * self.spacing_xy / 5, self.size * self.spacing_xy / 5))
        ax = fig.add_subplot(projection='3d')

        for x, y, z in self.storage_zone:
            ax.scatter(x, y, z, c='royalblue', s=80, alpha=0.8, edgecolors='w')
        for x, y, z in self.interaction_zone:
            ax.scatter(x, y, z, c='tomato', s=80, alpha=0.8, edgecolors='w')
        for x, y, z in self.readout_zone:
            ax.scatter(x, y, z, c='lightgreen', s=80, alpha=0.8, edgecolors='w')


        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", label="Storage Zone (L1-L5)", markerfacecolor="royalblue", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Interaction Zone (L1-L5)", markerfacecolor="tomato", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Readout Plane (L6)", markerfacecolor="lightgreen", markersize=10),
        ]
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1))

        ax.set_xlabel(r'X ($\mu m$)')
        ax.set_ylabel(r'Y ($\mu m$)')
        ax.set_zlabel(r'Z ($\mu m$)')

        ax.set_title(
            "Visualizing Atrium-style 3D NA Architecture"
        )

        ax.set_box_aspect(
            [self.size * self.spacing_xy, self.size * self.spacing_xy, self.layers * self.spacing_z]
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
    


    def get_available_3d_sites(self, initial_zone: str = "storage") -> List[Point3]:
        """
        Available 3D sites for initial placement.
        - storage: Only allow initial atoms to be in storage_zone (your requirement: all atoms start in storage)
        - all: Allow storage + interaction
        """
        zone = (initial_zone or "storage").lower()
        if zone == "storage":
            return sorted(set(self.storage_zone))
        if zone == "all":
            return sorted(set(self.storage_zone) | set(self.interaction_zone))
        raise ValueError(f"[Error] Unsupported initial_zone: {initial_zone!r}")

    def _default_architecture(self) -> Dict:
        """
        - trap grid size: size x size
        - trap spacing: spacing_xy
        - default all traps can place qubit (buffer is empty)
        """
        return {
            "atoms": {
                "N_x": int(self.size),
                "N_y": int(self.size),
                "site_separation": (float(self.spacing_xy), float(self.spacing_xy)),
            },
            "qubits": {
                "location": (0, 0),
                "N_spacing_atoms": (1, 1),
            },
        }

    def parse_slm_sites(self):
        """
        Configures SLM sites for storage and entanglement zones.
        """

        self.qubit_slm_sites = []
        self.buffer_slm_sites = []

        # read atoms
        N_x = int(self.architecture['atoms']['N_x'])
        N_y = int(self.architecture['atoms']['N_y'])
        site_separation_x = float(self.architecture['atoms']['site_separation'][0])
        site_separation_y = float(self.architecture['atoms']['site_separation'][1])

        for i in range(N_x):
            for j in range(N_y):
                if (i - self.architecture['qubits']['location'][0]) % self.architecture['qubits']['N_spacing_atoms'][0] == 0 and (j - self.architecture['qubits']['location'][1]) % self.architecture['qubits']['N_spacing_atoms'][1] == 0:
                    self.qubit_slm_sites.append((i * site_separation_x, j * site_separation_y))
                else:
                    self.buffer_slm_sites.append((i * site_separation_x, j * site_separation_y))

        # preserve determinism
        self.qubit_slm_sites = sorted(set(self.qubit_slm_sites))
        self.buffer_slm_sites = sorted(set(self.buffer_slm_sites))

    def _parse_qasm_lightweight(self, qasm_str: str) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Lightweight QASM parsing without relying on qiskit (supports common qreg/qbit declarations and two-qubit gate extraction).
        """
        # Strip comments
        lines = []
        for raw in qasm_str.splitlines():
            raw = raw.split("//", 1)[0].strip()
            if raw:
                lines.append(raw)

        n_qubits = 0
        # OPENQASM 2: qreg q[10];
        m2 = re.search(r"qreg\s+([a-zA-Z_]\w*)\s*\[\s*(\d+)\s*\]\s*;", qasm_str)
        if m2:
            n_qubits = int(m2.group(2))
        # OPENQASM 3: qubit[10] q;
        m3 = re.search(r"qubit\s*\[\s*(\d+)\s*\]\s*([a-zA-Z_]\w*)\s*;", qasm_str)
        if m3:
            n_qubits = max(n_qubits, int(m3.group(1)))

        if n_qubits <= 0:
            raise ValueError("[Error] Cannot detect number of qubits from QASM.")

        gates: List[Tuple[int, int]] = []
        # Extract any q[<idx>] occurrences per line.
        q_pat = re.compile(r"\bq\s*\[\s*(\d+)\s*\]")

        for ln in lines:
            l = ln.lower()
            if l.startswith(("openqasm", "include", "qreg", "creg", "qubit", "bit", "const", "defcal", "cal")):
                continue
            if "measure" in l or l.startswith("barrier"):
                continue

            qs = [int(x) for x in q_pat.findall(ln)]
            if len(qs) >= 2:
                q0, q1 = qs[0], qs[1]
                if q0 != q1:
                    gates.append((q0, q1))
                else:
                    gates.append((q0, q0))
            elif len(qs) == 1:
                q = qs[0]
                gates.append((q, q))

        return n_qubits, gates

    def set_program(self):
        """
        Sets up the quantum program by parsing the benchmark file and extracting gate information.
        """
        self.g_q = []
        with open(f"benchmark/{self.dir}/{self.benchmark}.{self.type}", 'r') as f:
            if self.type == "qasm":
                qasm_str = f.read()
                # Prefer qiskit if available, otherwise fallback to lightweight parsing.
                try:
                    from qiskit import QuantumCircuit, transpile  # type: ignore
                    try:
                        from qiskit.qasm3 import parse  # type: ignore
                    except Exception:
                        parse = None  # type: ignore

                    if "OPENQASM 2" in qasm_str:
                        circuit = QuantumCircuit.from_qasm_str(qasm_str)
                    elif "OPENQASM 3" in qasm_str and parse is not None:
                        circuit = parse(qasm_str)
                    else:
                        raise ImportError("Unsupported QASM version for qiskit parser.")

                    # Remove the last swap gates
                    while circuit.data and circuit.data[-1][0].name == "swap":
                        circuit.data.pop()

                    cz_circuit = transpile(
                        circuit,
                        basis_gates=["cz", "id", "u2", "u1", "u3"],
                        optimization_level=3,
                        seed_transpiler=0,
                    )
                    instruction = cz_circuit.data
                    self.results_code["n_qubits"] = cz_circuit.num_qubits
                    for inst in instruction:
                        if inst.operation.num_qubits == 2:
                            q0_idx = inst.qubits[0]._index
                            q1_idx = inst.qubits[1]._index
                            if q0_idx != q1_idx:
                                self.g_q.append((q0_idx, q1_idx))
                            elif inst.operation.name not in ["measure", "barrier", "id"]:
                                self.g_q.append((q0_idx, q0_idx))
                        elif inst.operation.name not in ["measure", "barrier"]:
                            self.g_q.append((inst.qubits[0]._index, inst.qubits[0]._index))
                except Exception:
                    n, gates = self._parse_qasm_lightweight(qasm_str)
                    self.results_code["n_qubits"] = n
                    self.g_q = gates
            elif self.type == "json":
                with open(f"benchmark/{self.dir}/{self.benchmark}.{self.type}", 'r') as f:
                    graphs = json.load(f)
                for q0, q1 in graphs:
                    self.results_code['n_qubits'] = max(self.results_code['n_qubits'], q0, q1)
                    if q0 == q1:
                        self.g_q.append((q0, q1))
                    else:
                        self.g_q.append((q0, q1))
                self.results_code['n_qubits'] += 1
            else:
                # Wait for the implementation of other file types
                raise ValueError("Unsupported benchmark file type.")

    def save_results(self):
        """
        Saves the results in JSON format within the specified directory.
        """
        output_dir = f"results/{self.results_code['dir']}/code/"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}{self.results_code['benchmark']}_code.json"
        with open(output_path, 'w') as f:
            json.dump(self.results_code, f, indent=2)
        print(f"[INFO] Atrium3D: Results saved to {output_path}")

    def solve(self, simulation: bool = False, animation: bool = False, initial_zone: str = "storage"):
        """
        Executes the compilation pipeline, including scheduling, routing, and optional simulation or animation.

        Args:
            simulation (bool): Whether to run a simulation.
            animation (bool): Whether to generate animations.
        """

        print(f"[INFO] Atrium3D: Start solving {self.benchmark}")
        # Prepare program + sites
        self.set_program()
        # 3D initial placement sites
        available_3d_sites = self.get_available_3d_sites(initial_zone=initial_zone)
        if self.results_code["n_qubits"] > len(available_3d_sites):
            raise ValueError("[Error] #qubits > #available 3D sites.")

        # Scheduling
        print(f"--------------------------------SCHEDULING--------------------------------")
        
        scheduler = Scheduler(
            results_code=self.results_code,
            g_q=self.g_q,
        )
        if self.scheduling_strategy == "asap":
            scheduler.asap()
            list_full_gates = scheduler.get_list_gates()
            self.results_code = scheduler.results_code
        else:
            raise ValueError(f"[Error] Unsupported scheduling strategy: {self.scheduling_strategy!r}")
        

        # Routing
        print(f"----------------------------PLACING & ROUTING----------------------------")
        router = Router(
            results_code=self.results_code,
            sites=[self.storage_zone, self.interaction_zone, self.readout_zone],
            list_full_gates=list_full_gates,
            initial_mapping=self.given_initial_mapping,
        )
        router.route_qubits()

        self.save_results()

        # Simulation
        if simulation:
            raise NotImplementedError("simulate() is not implemented in this repository.")

        # Animation
        if animation:
            animator = Animator(results_code=self.results_code)
            animator.animate()
        print(f"[INFO] Atrium3D: Finish solving {self.benchmark}\n")
        return self.results_code


