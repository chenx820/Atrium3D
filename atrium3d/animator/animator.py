import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

import os


class AODXLine:
    def __init__(self, y: float, z: float):
        self.y = y
        self.z = z

    def update_position(self, y: float, z: float):
        self.y = y
        self.z = z

class AODYLine:
    def __init__(self, x: float, z: float):
        self.x = x
        self.z = z

    def update_position(self, x: float, z: float):
        self.x = x
        self.z = z

class AODs:
    def __init__(self, n_x_aods: int = 10, n_y_aods: int = 10):
        self.n_x_aods = n_x_aods
        self.n_y_aods = n_y_aods
        self.aod_x_lines = [AODXLine(0, 0) for _ in range(n_x_aods)]
        self.aod_y_lines = [AODYLine(0, 0) for _ in range(n_y_aods)]


class Animator:            
    def __init__(self, compiler, fps: int = 15):
        self.compiler = compiler
        self.fps = fps
        self.aod_x_lines = AODs()
        self.aod_y_lines = AODs()

    def generate_animation(self, compiler):
        """
        Generate and save the 3D animation of atomic shuttling.
        """
        stage_positions = compiler.results_code.get("stage_positions", [])
        stage_meta = compiler.results_code.get("stage_placement_meta", [])
        steps_per_move = int(getattr(compiler, "routing_steps_per_move", 15))
        pause_frames = int(getattr(compiler, "routing_pause_frames", 5))
        n_stages = len(stage_positions)
        frames_per_segment = (steps_per_move + pause_frames) if n_stages > 1 else 0
        z_laser_top = (compiler.layers - 1) * compiler.spacing_z  # laser from readout plane (z) down to atom
        if not stage_positions:
            raise ValueError("[Error] No stage_positions found. Make sure to run solve() first.")
        if n_stages < 2:
            raise ValueError("[Error] Need at least 2 stages to animate (stage_positions length < 2).")

        output_dir = f"results/{compiler.results_code['dir']}/animation/"
        os.makedirs(output_dir, exist_ok=True)
        save_path = output_dir + compiler.benchmark + ".mp4"

        fig = plt.figure(figsize=(10, 12))
        ax = fig.add_subplot(projection="3d")

        compiler._plot_architecture_background(ax, alpha_storage=0.1, alpha_interaction=0.1, alpha_readout=0.1)

        n_qubits = compiler.results_code["n_qubits"]

        # Per-atom AOD lines: x-AOD (∥x, moves in yz) and y-AOD (∥y, moves in xz), follow each atom
        aod_x_lines = []
        aod_y_lines = []
        for _ in range(n_qubits):
            (lx,) = ax.plot([0, 0], [0, 0], [0, 0], color="deepskyblue", alpha=0.5, linewidth=2.0)
            (ly,) = ax.plot([0, 0], [0, 0], [0, 0], color="darkorange", alpha=0.5, linewidth=2.0)
            aod_x_lines.append(lx)
            aod_y_lines.append(ly)

        # Individual addressing lasers (from +z down to atom) when gate is applied
        ia_laser_lines = []
        for _ in range(n_qubits):
            (ll,) = ax.plot([0, 0], [0, 0], [0, 0], color="gold", alpha=0.85, linewidth=2.5)
            ia_laser_lines.append(ll)

        scatter = ax.scatter([], [], [], c="#d62728", s=100, edgecolors="white", linewidths=1.0, alpha=0.9)
        
        texts = [ax.text(0, 0, 0, str(q), fontsize=8, color="black", weight='bold') 
                for q in range(n_qubits)]

        aod_legend = [
            Line2D([0], [0], color="deepskyblue", linewidth=2.5, alpha=0.8, label="AOD ∥ x (moves in yz)"),
            Line2D([0], [0], color="darkorange", linewidth=2.5, alpha=0.8, label="AOD ∥ y (moves in xz)"),
            Line2D([0], [0], color="gold", linewidth=2.5, alpha=0.85, label="Individual addressing (gate)"),
        ]
        ax.legend(handles=aod_legend, loc="upper left", bbox_to_anchor=(0.8, 1))

        ax.set_xlabel(r"X ($\mu m$)")
        ax.set_ylabel(r"Y ($\mu m$)")
        ax.set_zlabel(r"Z ($\mu m$)")
        ax.set_box_aspect([
            compiler.size * compiler.spacing_xy, 
            compiler.size * compiler.spacing_xy, 
            compiler.layers * compiler.spacing_z
        ])

        x_AOD_min = -0.5 * compiler.spacing_xy
        x_AOD_max = compiler.size * compiler.spacing_xy + 0.5 * compiler.spacing_xy
        y_AOD_min = -0.5 * compiler.spacing_xy
        y_AOD_max = compiler.size * compiler.spacing_xy + 0.5 * compiler.spacing_xy

        _eps = 1e-6  # position change threshold (µm)

        total_frames = (n_stages - 1) * frames_per_segment

        def _frame_positions(frame_idx: int) -> dict:
            """
            根据 stage_positions 在动画帧索引上做插值，动态计算每一帧的位置。
            注意：不把逐帧数据写入 results_code，避免文件过大。
            """
            seg = frame_idx // frames_per_segment
            pos_in_seg = frame_idx % frames_per_segment
            if seg >= n_stages - 1:
                return stage_positions[-1]

            start_map = stage_positions[seg]
            end_map = stage_positions[seg + 1]

            # Gate(pause) frames: hold at end_map.
            if pos_in_seg >= steps_per_move:
                return end_map

            alpha = pos_in_seg / float(max(1, steps_per_move))
            out = {}
            for q in range(n_qubits):
                qs = str(q)
                s = start_map[qs]
                e = end_map[qs]
                out[qs] = [
                    s[0] * (1 - alpha) + e[0] * alpha,
                    s[1] * (1 - alpha) + e[1] * alpha,
                    s[2] * (1 - alpha) + e[2] * alpha,
                ]
            return out

        def update(frame_idx):
            current_positions = _frame_positions(frame_idx)
            prev_positions = _frame_positions(frame_idx - 1) if frame_idx > 0 else None
            xs, ys, zs = [], [], []
            
            for q in range(n_qubits):
                pos = current_positions[str(q)]
                x, y, z = pos[0], pos[1], pos[2]
                xs.append(x)
                ys.append(y)
                zs.append(z)
                
                texts[q].set_position((x + 0.5, y + 0.5))
                texts[q].set_3d_properties(z + 0.5, "z")

                # AOD only when this atom moved (AOD is "holding" it during shuttle)
                if prev_positions is not None:
                    prev = prev_positions[str(q)]
                    moved = (
                        abs(prev[0] - x) > _eps
                        or abs(prev[1] - y) > _eps
                        or abs(prev[2] - z) > _eps
                    )
                else:
                    moved = False

                if moved:
                    aod_x_lines[q].set_visible(True)
                    aod_x_lines[q].set_data([x_AOD_min, x_AOD_max], [y, y])
                    aod_x_lines[q].set_3d_properties([z, z])
                    aod_y_lines[q].set_visible(True)
                    aod_y_lines[q].set_data([x, x], [y_AOD_min, y_AOD_max])
                    aod_y_lines[q].set_3d_properties([z, z])
                else:
                    aod_x_lines[q].set_visible(False)
                    aod_y_lines[q].set_visible(False)

            # Individual addressing lasers: only during gate (pause) phase, on qubits in two_qubit_gates
            active_qubits_ia = set()
            if frames_per_segment > 0 and n_stages > 1 and stage_meta:
                segment = frame_idx // frames_per_segment
                pos_in_seg = frame_idx % frames_per_segment
                if pos_in_seg >= steps_per_move and segment + 1 < len(stage_meta):
                    micro_stage = segment + 1
                    meta = stage_meta[micro_stage]
                    for gate in meta.get("two_qubit_gates", []):
                        if len(gate) >= 2:
                            active_qubits_ia.add(int(gate[0]))
                            active_qubits_ia.add(int(gate[1]))
            for q in range(n_qubits):
                if q in active_qubits_ia:
                    x, y, z = current_positions[str(q)][0], current_positions[str(q)][1], current_positions[str(q)][2]
                    ia_laser_lines[q].set_data([x, x], [y, y])
                    ia_laser_lines[q].set_3d_properties([z_laser_top, z])
                    ia_laser_lines[q].set_visible(True)
                else:
                    ia_laser_lines[q].set_visible(False)

            scatter._offsets3d = (xs, ys, zs)
            ax.set_title(f"3D Atom Shuttling: {compiler.benchmark}\nFrame {frame_idx}/{total_frames}")
            
            return scatter, *texts, *aod_x_lines, *aod_y_lines, *ia_laser_lines

        print(f"[INFO] Rendering {total_frames} frames to {save_path}... This may take a moment.")
        
        anim = FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=False)
        

        anim.save(save_path, writer='ffmpeg', fps=self.fps, dpi=150, extra_args=['-vcodec', 'libx264'])
        
        plt.close(fig)
        print(f"[INFO] Animation successfully saved to: {save_path}")