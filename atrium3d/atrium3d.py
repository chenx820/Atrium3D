import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from atrium3d.scheduler.scheduler import scheduling
from atrium3d.placer import place_stages, placing_3d
from atrium3d.router import routing
from atrium3d.animator import generate_animation


class Atrium3D:
    def __init__(
        self,
        benchmark: str = "qft_n10",
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
        self.qubit_slm_sites: List[Tuple[float, float]] = []
        self.buffer_slm_sites: List[Tuple[float, float]] = []

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
    
    def visualize_initial_mapping(self, mapping=None, save_path: Optional[str] = None, show: bool = False):
        """
        Visualize the initial mapping.
        Args:
            mapping: The initial mapping.
            save_path: The path to save the image.
            show: Whether to show the image.
        """
        if mapping is None:
            mapping = self.results_code.get("initial_mapping")
        if not mapping:
            raise ValueError("[Error] initial_mapping not found. Please run solve() first.")

        # Determine dimensionality
        first_key = next(iter(mapping))
        first_val = mapping[first_key]
        dim = len(first_val)

        if dim == 2:
            if not self.qubit_slm_sites and not self.buffer_slm_sites:
                self.parse_slm_sites()

            fig, ax = plt.subplots(figsize=(8, 8))
            if self.qubit_slm_sites:
                xs, ys = zip(*self.qubit_slm_sites)
                ax.scatter(xs, ys, s=35, c="#c7c7c7", alpha=0.7, edgecolors="none", label="Qubit SLM sites")
            if self.buffer_slm_sites:
                bx, by = zip(*self.buffer_slm_sites)
                ax.scatter(bx, by, s=35, facecolors="none", edgecolors="#2ca02c", linewidths=1.0, alpha=0.6, label="Buffer sites")

            mx = []
            my = []
            for q in sorted(mapping.keys()):
                x, y = mapping[q]
                mx.append(float(x))
                my.append(float(y))
            ax.scatter(mx, my, s=110, c="#d62728", alpha=0.9, edgecolors="white", linewidths=0.8, label="Initial mapping")
            for q in sorted(mapping.keys()):
                x, y = mapping[q]
                ax.text(float(x) + 0.2, float(y) + 0.2, str(q), fontsize=9, color="black")

            ax.set_title(f"Initial Mapping (2D SLM) benchmark={self.benchmark}, n_qubits={self.results_code.get('n_qubits', '?')}")
            ax.set_xlabel(r"X ($\mu m$)")
            ax.set_ylabel(r"Y ($\mu m$)")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle="--", alpha=0.25)
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)
            return

        if dim != 3:
            raise ValueError(f"[Error] Unsupported mapping dimensionality: {dim}")

        # 3D mapping on Atrium architecture
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(projection="3d")

        # Background architecture points
        if self.storage_zone:
            sx, sy, sz = zip(*sorted(self.storage_zone))
            ax.scatter(sx, sy, sz, c="royalblue", s=30, alpha=0.25, edgecolors="none", label="Storage zone")
        if self.interaction_zone:
            ix, iy, iz = zip(*sorted(self.interaction_zone))
            ax.scatter(ix, iy, iz, c="tomato", s=35, alpha=0.25, edgecolors="none", label="Interaction zone")
        if self.readout_zone:
            rx, ry, rz = zip(*sorted(self.readout_zone))
            ax.scatter(rx, ry, rz, c="lightgreen", s=25, alpha=0.2, edgecolors="none", label="Readout plane")

        # Highlight mapping
        mx, my, mz = [], [], []
        for q in sorted(mapping.keys()):
            x, y, z = mapping[q]
            mx.append(float(x))
            my.append(float(y))
            mz.append(float(z))
        ax.scatter(mx, my, mz, c="#d62728", s=140, alpha=0.95, edgecolors="white", linewidths=0.8, label="Initial mapping (3D)")
        for q in sorted(mapping.keys()):
            x, y, z = mapping[q]
            ax.text(float(x) + 0.5, float(y) + 0.5, float(z) + 0.5, str(q), fontsize=9, color="black")

        ax.set_xlabel(r"X ($\mu m$)")
        ax.set_ylabel(r"Y ($\mu m$)")
        ax.set_zlabel(r"Z ($\mu m$)")
        ax.set_title(f"Initial Mapping on 3D Atrium (benchmark={self.benchmark}, n_qubits={self.results_code.get('n_qubits', '?')})")
        ax.set_box_aspect([self.size * self.spacing_xy, self.size * self.spacing_xy, self.layers * self.spacing_z])
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def _plot_architecture_background(self, ax, alpha_storage: float = 0.18, alpha_interaction: float = 0.18, alpha_readout: float = 0.12):
        """
        Plot the architecture background.
        Args:
            ax: The axis to plot on.
            alpha_storage: The alpha value for the storage zone.
            alpha_interaction: The alpha value for the interaction zone.
            alpha_readout: The alpha value for the readout zone.
        """
        if self.storage_zone:
            sx, sy, sz = zip(*sorted(self.storage_zone))
            ax.scatter(sx, sy, sz, c="royalblue", s=18, alpha=alpha_storage, edgecolors="none", label="Storage zone")
        if self.interaction_zone:
            ix, iy, iz = zip(*sorted(self.interaction_zone))
            ax.scatter(ix, iy, iz, c="tomato", s=20, alpha=alpha_interaction, edgecolors="none", label="Interaction zone")
        if self.readout_zone:
            rx, ry, rz = zip(*sorted(self.readout_zone))
            ax.scatter(rx, ry, rz, c="lightgreen", s=16, alpha=alpha_readout, edgecolors="none", label="Readout plane")

    def get_available_3d_sites(self, initial_zone: str = "storage") -> List[Tuple[float, float, float]]:
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

    def solve(self, simulation: bool = False, animation: bool = False, do_routing: bool = False, initial_zone: str = "storage"):
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
        print(f"[INFO] Atrium3D: Start {self.scheduling_strategy.upper()} scheduling")
        self.results_code, list_scheduling = scheduling(g_q=self.g_q, results_code=self.results_code, scheduling_strategy=self.scheduling_strategy)        
        print("[INFO] Atrium3D: Scheduling finished")
        
        list_gate = []
        for gates in list_scheduling:
            tmp = [self.g_q[i] for i in gates]
            list_gate.append(tmp)

        # Readout urgency heuristic (two-qubit last-use):
        # If a qubit finishes earlier in terms of its LAST TWO-QUBIT gate, we prefer placing it closer to readout plane
        # so that moving it to readout after completion is faster / less blocking.
        n_qubits = int(self.results_code.get("n_qubits", 0))
        last_stage_2q = [-1 for _ in range(n_qubits)]
        for s_idx, stage in enumerate(list_gate):
            for q0, q1 in stage:
                # Only count two-qubit gates.
                if q0 == q1:
                    continue
                if 0 <= q0 < n_qubits:
                    last_stage_2q[q0] = max(last_stage_2q[q0], s_idx)
                if 0 <= q1 < n_qubits:
                    last_stage_2q[q1] = max(last_stage_2q[q1], s_idx)

        n_stages = max(1, int(self.results_code.get("n_stages", len(list_gate))))
        # urgency in [0,1], higher means earlier-finished -> more urgent to be near readout
        readout_urgency = []
        for q in range(n_qubits):
            ls = last_stage_2q[q]
            if ls < 0:
                # never used in two-qubit gates -> can be treated as "finish immediately"
                readout_urgency.append(1.0)
            else:
                readout_urgency.append(float((n_stages - 1 - ls) / max(1, n_stages - 1)))

        readout_plane_z = float((self.layers - 1) * self.spacing_z)

        # Initial mapping
        print("[INFO] Atrium3D: Start initial mapping")
        # All atoms start in storage_zone: no longer biased towards interaction. Routing/moving to interaction will be done later when actually performing gates.
        preferred = sorted(set(self.storage_zone))
        self.results_code, best_mapping = placing_3d(
            available_sites=available_3d_sites,
            preferred_sites=preferred,
            results_code=self.results_code,
            list_full_gates=list_gate,
            qubits_mapping=self.given_initial_mapping,
            readout_plane_z=readout_plane_z,
            readout_urgency=readout_urgency,
            readout_weight=float(self.results_code.get("readout_weight", 0.0)),
        )
        self.results_code["initial_zone"] = initial_zone
        self.results_code["readout_urgency"] = readout_urgency
        self.results_code["readout_last_stage_2q"] = last_stage_2q
        self.results_code["readout_urgency_basis"] = "last_two_qubit_gate"
        self.results_code["readout_plane_z"] = readout_plane_z
        print("[INFO] Atrium3D: Initial mapping finished. Best mapping: ", best_mapping)

        # Stage-by-stage placement (given scheduling)
        print("[INFO] Atrium3D: Start stage placement")
        stage_positions, stage_meta, stage_summary = place_stages(
            initial_mapping=best_mapping,
            stages=list_gate,
            size=self.size,
            layers=self.layers,
            center_range=self.center_range,
            spacing_xy=float(self.spacing_xy),
            spacing_z=float(self.spacing_z),
            last_stage_2q=last_stage_2q,
            enable_readout_move=True,
        )
        # JSON-friendly serialization
        self.results_code["stage_positions"] = [
            {str(q): [p[0], p[1], p[2]] for q, p in stage_map.items()} for stage_map in stage_positions
        ]
        self.results_code["stage_placement_meta"] = stage_meta
        self.results_code["stage_placement_summary"] = stage_summary
        print("[INFO] Atrium3D: Stage placement finished")

        # Routing
        if do_routing:
            t0 = time.perf_counter()
            self.results_code = routing(self.results_code)
            elapsed = time.perf_counter() - t0
            ct = self.results_code.setdefault("compilation_time", {})
            ct["routing"] = float(elapsed)
            ct["total"] = float(ct.get("total", 0.0) + elapsed)

        self.save_results()

        # Simulation
        if simulation:
            raise NotImplementedError("simulate() is not implemented in this repository.")

        # Animation
        if animation:
            # If user explicitly requests routing but directly requires animation, run routing again automatically
            if "routing_frames" not in self.results_code:
                self.results_code = routing(self.results_code)
            generate_animation(self)
        print(f"[INFO] Atrium3D: Finish solving {self.benchmark}\n")
        return self.results_code


