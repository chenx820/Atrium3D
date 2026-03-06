from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from atrium3d.scheduler.scheduler import scheduling
from atrium3d.placer import place_stages, placing_3d, placing_slm


class Atom:
    def __init__(self, atom_id, grid_pos):
        self.id = atom_id
        self.grid_pos = grid_pos

class Atrium3D:
    def __init__(
        self,
        benchmark: str = "qft_n10",
        dir: str = "default",
        type: str = "qasm",
        size: int = 10,
        layers: int = 4,
        architecture: Optional[Dict] = None,
        scheduling_strategy: str = "asap",
        given_initial_mapping=None,
    ):
        self.benchmark = benchmark
        self.dir = dir
        self.type = type
        self.size = size
        self.layers = layers
        self.scheduling_strategy = scheduling_strategy
        self.given_initial_mapping = given_initial_mapping
        self.grid = {}
        self.center_range = range(2, self.size - 2) # center 3x3 is the interaction zone
        self.spacing_xy = 5.0
        self.spacing_z = 25.0

        # safety radius
        self.r_route = 2.0
        self.r_gate = 12.0  # 近似 Rydberg blockade 半径 (um)

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
                    self.grid[(x, y, z)] = Atom(f"{zone}_{x}_{y}_{z}", (x, y, z))

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
        可视化 `initial_mapping`。
        - 若映射是 (x,y)：画 2D SLM trap
        - 若映射是 (x,y,z)：画 3D Atrium 架构并高亮 qubit 初始位置
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
        if self.storage_zone:
            sx, sy, sz = zip(*sorted(self.storage_zone))
            ax.scatter(sx, sy, sz, c="royalblue", s=18, alpha=alpha_storage, edgecolors="none", label="Storage zone")
        if self.interaction_zone:
            ix, iy, iz = zip(*sorted(self.interaction_zone))
            ax.scatter(ix, iy, iz, c="tomato", s=20, alpha=alpha_interaction, edgecolors="none", label="Interaction zone")
        if self.readout_zone:
            rx, ry, rz = zip(*sorted(self.readout_zone))
            ax.scatter(rx, ry, rz, c="lightgreen", s=16, alpha=alpha_readout, edgecolors="none", label="Readout plane")

    def visualize_micro_stage(self, micro_stage_idx: int, save_path: Optional[str] = None, show: bool = False, dpi: int = 250):
        """
        可视化一个 micro-stage 的全局原子位置（来自 results_code['stage_positions']），并高亮本 stage 的 gate 落点与 readout move。
        """
        stage_positions = self.results_code.get("stage_positions")
        stage_meta = self.results_code.get("stage_placement_meta")
        if not stage_positions or not stage_meta:
            raise ValueError("[Error] stage placement not found. Please run solve() first.")

        if micro_stage_idx < 0 or micro_stage_idx >= len(stage_positions):
            raise ValueError(f"[Error] micro_stage_idx out of range: {micro_stage_idx}")

        pos_map_raw = stage_positions[micro_stage_idx]
        # JSON stores keys as str -> [x,y,z]
        pos_map: Dict[int, Tuple[float, float, float]] = {int(q): (float(p[0]), float(p[1]), float(p[2])) for q, p in pos_map_raw.items()}
        meta = stage_meta[micro_stage_idx]

        active_qubits = set()
        for g in meta.get("two_qubit_gates", []):
            if len(g) == 2:
                active_qubits.add(int(g[0]))
                active_qubits.add(int(g[1]))

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(projection="3d")

        self._plot_architecture_background(ax)

        # Plot qubits
        inactive_x, inactive_y, inactive_z = [], [], []
        active_x, active_y, active_z = [], [], []
        for q in sorted(pos_map.keys()):
            x, y, z = pos_map[q]
            if q in active_qubits:
                active_x.append(x); active_y.append(y); active_z.append(z)
            else:
                inactive_x.append(x); inactive_y.append(y); inactive_z.append(z)

        if inactive_x:
            ax.scatter(inactive_x, inactive_y, inactive_z, c="#7f7f7f", s=45, alpha=0.55, edgecolors="white", linewidths=0.4, label="Qubits (inactive)")
        if active_x:
            ax.scatter(active_x, active_y, active_z, c="#d62728", s=95, alpha=0.95, edgecolors="white", linewidths=0.8, label="Qubits (active)")

        # Label active qubits only (avoid clutter)
        for q in sorted(active_qubits):
            if q in pos_map:
                x, y, z = pos_map[q]
                ax.text(x + 0.5, y + 0.5, z + 0.5, str(q), fontsize=9, color="black")

        # Draw gate site links in interaction zone
        for gs in meta.get("gate_sites", []):
            pts = gs.get("sites_phys")
            if not pts or len(pts) != 2:
                continue
            (x0, y0, z0), (x1, y1, z1) = (pts[0], pts[1])
            ax.plot([x0, x1], [y0, y1], [z0, z1], c="#ffbf00", linewidth=3, alpha=0.9)
            ax.scatter([x0, x1], [y0, y1], [z0, z1], c="#ffbf00", s=55, alpha=0.95, edgecolors="black", linewidths=0.4)

        # Draw readout moves
        for mv in meta.get("readout_moves", []):
            fr = mv.get("from")
            to = mv.get("to")
            if not fr or not to:
                continue
            ax.plot([fr[0], to[0]], [fr[1], to[1]], [fr[2], to[2]], c="#2ca02c", linewidth=2.5, alpha=0.9)

        ax.set_xlabel(r"X ($\mu m$)")
        ax.set_ylabel(r"Y ($\mu m$)")
        ax.set_zlabel(r"Z ($\mu m$)")
        ax.set_box_aspect([self.size * self.spacing_xy, self.size * self.spacing_xy, self.layers * self.spacing_z])

        title = f"Micro-stage {meta.get('micro_stage', micro_stage_idx)} (orig stage={meta.get('original_stage','?')}, travel={meta.get('travel_distance',0):.2f})"
        ax.set_title(title)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

        if save_path:
            plt.savefig(save_path, dpi=int(dpi), bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def save_all_micro_stage_images(
        self,
        out_dir: str | Path,
        prefix: str = "stage_",
        every: int = 1,
        max_frames: Optional[int] = None,
        dpi: int = 250,
    ) -> List[str]:
        """
        导出所有 micro-stage 的 PNG 图到 out_dir。
        返回生成的文件路径列表。
        """
        stage_positions = self.results_code.get("stage_positions")
        stage_meta = self.results_code.get("stage_placement_meta")
        if not stage_positions or not stage_meta:
            raise ValueError("[Error] stage placement not found. Please run solve() first.")

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        n = len(stage_positions)
        step = max(1, int(every))
        limit = n if max_frames is None else min(n, int(max_frames))

        generated: List[str] = []
        for i in range(0, limit, step):
            fname = f"{prefix}{i:04d}.png"
            fpath = str(out_path / fname)
            self.visualize_micro_stage(i, save_path=fpath, show=False, dpi=dpi)
            generated.append(fpath)
        return generated

    def get_available_3d_sites(self, initial_zone: str = "storage") -> List[Tuple[float, float, float]]:
        """
        初始放置可用的 3D sites。
        - storage: 只允许初始原子位于 storage_zone（你的需求：刚开始都在 storage）
        - all: 允许 storage + interaction
        """
        zone = (initial_zone or "storage").lower()
        if zone == "storage":
            return sorted(set(self.storage_zone))
        if zone == "all":
            return sorted(set(self.storage_zone) | set(self.interaction_zone))
        raise ValueError(f"[Error] Unsupported initial_zone: {initial_zone!r}")

    def _default_architecture(self) -> Dict:
        """
        当前仓库里没有 architecture/ 配置文件，因此提供一个与本项目目录结构相适配的默认架构：
        - trap 网格大小: size x size
        - trap 间距: spacing_xy
        - 默认所有 trap 都可放 qubit（buffer 为空）
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
        不依赖 qiskit 的轻量 QASM 解析（支持常见的 qreg/qbit 声明与两比特门提取）。
        目标是为 scheduling+placing 提供 gate 列表，而不是完整语义执行。
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
        print(f"[INFO] BAM: Results saved to {output_path}")

    def solve(self, simulation: bool = False, animation: bool = False, do_routing: bool = False, initial_zone: str = "storage"):
        """
        Executes the compilation pipeline, including scheduling, routing, and optional simulation or animation.

        Args:
            simulation (bool): Whether to run a simulation.
            animation (bool): Whether to generate animations.
        """

        print(f"[INFO] BAM: Start solving {self.benchmark}")
        # Prepare program + sites
        self.set_program()
        # 3D initial placement sites
        available_3d_sites = self.get_available_3d_sites(initial_zone=initial_zone)
        if self.results_code["n_qubits"] > len(available_3d_sites):
            raise ValueError("[Error] #qubits > #available 3D sites.")

        # Scheduling
        print(f"[INFO] BAM: Start {self.scheduling_strategy.upper()} scheduling")
        self.results_code, list_scheduling = scheduling(g_q=self.g_q, results_code=self.results_code, scheduling_strategy=self.scheduling_strategy)        
        print("[INFO] BAM: Scheduling finished")
        
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
        print("[INFO] BAM: Start initial mapping")
        # 初始都在 storage_zone：不再偏向 interaction。后续真正做门时再路由/移动到 interaction。
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
        print("[INFO] BAM: Initial mapping finished. Best mapping: ", best_mapping)

        # Stage-by-stage placement (given scheduling)
        print("[INFO] BAM: Start stage placement")
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
        print("[INFO] BAM: Stage placement finished")

        # Routing
        if do_routing:
            raise NotImplementedError(
                "routing() 目前未在该仓库实现。你想要的下一步是新增 routing 模块时，我可以继续补齐。"
            )

        self.save_results()

        # Simulation
        if simulation:
            raise NotImplementedError("simulate() 目前未在该仓库实现。")

        # Animation
        if animation:
            raise NotImplementedError("animate() 目前未在该仓库实现。")
        print(f"[INFO] BAM: Finish solving {self.benchmark}\n")
        return self.results_code


